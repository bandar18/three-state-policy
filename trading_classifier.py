#!/usr/bin/env python3
"""
Multi-Asset Trading Classification System
========================================

A supervised learning system for predicting SPY returns using multi-asset features.
Implements Bull/Bear/Flat classification with risk management and performance evaluation.
"""

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from typing import Dict, Tuple, List, Optional
# import talib  # Commented out for compatibility


class TradingClassifier:
    """
    Multi-Asset Trading Classification System
    """
    
    def __init__(self):
        self.feature_window = 75  # minutes
        self.prediction_horizon = 15  # minutes
        self.decision_cadence = 5  # minutes
        self.return_threshold = 0.0002  # 0.02% (0.05 × average bid-ask spread)
        self.backtest_days = 60
        self.vol_filter_threshold = 25  # VIX threshold
        self.max_drawdown_limit = 0.05  # 5%
        self.scaler = StandardScaler()
        
        # Create figures directory
        os.makedirs('./figs', exist_ok=True)
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_atr(self, high, low, close, period=20):
        """Calculate ATR (Average True Range) indicator"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def load_data(self) -> pd.DataFrame:
        """
        Load SPY, VIX, and DXY data from yfinance
        """
        print("Loading market data...")
        
        # Download data for last 7 days (Yahoo Finance limit for 1-minute data)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Maximum allowed for 1-minute data
        
        # Download main instruments
        spy = yf.download('SPY', start=start_date, end=end_date, interval='1m', progress=False)
        vix = yf.download('^VIX', start=start_date, end=end_date, interval='1m', progress=False)
        dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, interval='1m', progress=False)
        
        # Clean and align data
        spy = spy.dropna()
        vix = vix.dropna()
        dxy = dxy.dropna()
        
        # Align timestamps and forward fill missing values
        common_index = spy.index.intersection(vix.index).intersection(dxy.index)
        
        if len(common_index) == 0:
            raise ValueError("No common timestamps found between instruments")
        
        spy = spy.loc[common_index]
        vix = vix.loc[common_index]
        dxy = dxy.loc[common_index]
        
        # Create combined dataset
        data = pd.DataFrame(index=common_index)
        data['SPY_Open'] = spy['Open']
        data['SPY_High'] = spy['High']
        data['SPY_Low'] = spy['Low']
        data['SPY_Close'] = spy['Close']
        data['SPY_Volume'] = spy['Volume']
        data['VIX_Close'] = vix['Close']
        data['DXY_Close'] = dxy['Close']
        
        # Forward fill any remaining NaNs
        data = data.ffill().dropna()
        
        print(f"Loaded {len(data)} 1-minute bars")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
    
    def build_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build comprehensive feature set for each 5-minute bar
        """
        print("Building features...")
        
        # Resample to 5-minute bars
        ohlcv_dict = {
            'SPY_Open': 'first',
            'SPY_High': 'max',
            'SPY_Low': 'min',
            'SPY_Close': 'last',
            'SPY_Volume': 'sum',
            'VIX_Close': 'last',
            'DXY_Close': 'last'
        }
        
        df = data.resample('5T').agg(ohlcv_dict).dropna()
        
        # Calculate basic returns
        df['SPY_Return_1'] = df['SPY_Close'].pct_change(1) * 100  # 1-period return
        df['SPY_Return_5'] = df['SPY_Close'].pct_change(5) * 100  # 5-period return
        df['SPY_Return_15'] = df['SPY_Close'].pct_change(15) * 100  # 15-period return
        
        # Rolling volatility (10 periods = 50 minutes)
        df['SPY_Volatility'] = df['SPY_Return_1'].rolling(10).std()
        
        # Realized skewness (30 periods = 150 minutes)
        df['SPY_Skew'] = df['SPY_Return_1'].rolling(30).skew()
        
        # Volume z-score (20 periods = 100 minutes)
        df['Volume_ZScore'] = (df['SPY_Volume'] - df['SPY_Volume'].rolling(20).mean()) / df['SPY_Volume'].rolling(20).std()
        
        # Technical indicators (simple implementations)
        # RSI (14 periods)
        df['RSI'] = self._calculate_rsi(df['SPY_Close'], 14)
        
        # MACD
        macd, macd_signal, macd_hist = self._calculate_macd(df['SPY_Close'], 12, 26, 9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # ATR (20 periods)
        df['ATR'] = self._calculate_atr(df['SPY_High'], df['SPY_Low'], df['SPY_Close'], 20)
        
        # Cross-asset features
        df['VIX_Return'] = df['VIX_Close'].pct_change() * 100
        df['DXY_Return'] = df['DXY_Close'].pct_change() * 100
        df['SPY_VIX_Spread'] = df['SPY_Return_1'] - df['VIX_Return']
        df['SPY_DXY_Spread'] = df['SPY_Return_1'] - df['DXY_Return']
        
        # Feature list for stacking
        feature_cols = [
            'SPY_Return_1', 'SPY_Return_5', 'SPY_Return_15',
            'SPY_Volatility', 'SPY_Skew', 'Volume_ZScore',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR',
            'VIX_Return', 'DXY_Return', 'SPY_VIX_Spread', 'SPY_DXY_Spread'
        ]
        
        # Create stacked features (last 75 minutes = 15 periods × 15 features)
        periods_to_stack = 15  # 75 minutes / 5 minutes
        
        stacked_features = []
        for i in range(periods_to_stack, len(df)):
            row_features = []
            for period in range(periods_to_stack):
                idx = i - period
                for col in feature_cols:
                    if pd.notna(df.iloc[idx][col]):
                        row_features.append(df.iloc[idx][col])
                    else:
                        row_features.append(0.0)
            stacked_features.append(row_features)
        
        # Create feature matrix
        feature_names = []
        for period in range(periods_to_stack):
            for col in feature_cols:
                feature_names.append(f"{col}_lag_{period}")
        
        features_df = pd.DataFrame(stacked_features, 
                                 index=df.index[periods_to_stack:], 
                                 columns=feature_names)
        
        # Add current period basic info
        features_df['Current_VIX'] = df['VIX_Close'].iloc[periods_to_stack:].values
        features_df['Current_ATR'] = df['ATR'].iloc[periods_to_stack:].values
        features_df['Current_Close'] = df['SPY_Close'].iloc[periods_to_stack:].values
        
        print(f"Built {len(features_df)} feature rows with {len(features_df.columns)} features")
        
        return features_df
    
    def make_labels(self, data: pd.DataFrame) -> pd.Series:
        """
        Create classification labels based on future returns
        """
        print("Creating labels...")
        
        # Calculate future 15-minute aggregate return
        future_returns = []
        for i in range(len(data) - 3):  # 3 periods = 15 minutes
            current_close = data['Current_Close'].iloc[i]
            future_close = data['Current_Close'].iloc[i + 3]
            future_return = (future_close / current_close) - 1
            future_returns.append(future_return)
        
        # Pad with NaN for the last 3 periods
        future_returns.extend([np.nan] * 3)
        
        # Create labels
        labels = []
        for ret in future_returns:
            if pd.isna(ret):
                labels.append(np.nan)
            elif ret >= self.return_threshold:
                labels.append(1)  # Bull
            elif ret <= -self.return_threshold:
                labels.append(-1)  # Bear
            else:
                labels.append(0)  # Flat
        
        labels_series = pd.Series(labels, index=data.index, name='Label')
        
        print(f"Label distribution:")
        print(f"Bull: {sum(l == 1 for l in labels if not pd.isna(l))}")
        print(f"Bear: {sum(l == -1 for l in labels if not pd.isna(l))}")
        print(f"Flat: {sum(l == 0 for l in labels if not pd.isna(l))}")
        
        return labels_series
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostingClassifier:
        """
        Train gradient boosting classifier with time series cross-validation
        """
        print("Training model...")
        
        # Remove NaN labels
        mask = ~pd.isna(y)
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Handle NaN values in features
        X_clean = X_clean.fillna(X_clean.mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        # Grid search with time series CV
        gb_classifier = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(
            gb_classifier, 
            param_grid, 
            cv=tscv, 
            scoring='balanced_accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y_clean)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return best_model
    
    def walk_forward_backtest(self, data: pd.DataFrame, labels: pd.Series, 
                            model: GradientBoostingClassifier) -> Dict:
        """
        Perform walk-forward backtesting with expanding window
        """
        print("Running walk-forward backtest...")
        
        # Prepare data
        feature_cols = [col for col in data.columns if col not in ['Current_VIX', 'Current_ATR', 'Current_Close']]
        
        # Split data for walk-forward testing (adjusted for 7-day limit)
        total_samples = len(data)
        train_samples = int(total_samples * 0.6)  # 60% for training
        test_samples = int(total_samples * 0.2)   # 20% for testing
        
        results = []
        trades = []
        
        # Walk-forward loop (simplified for demo with limited data)
        for offset in range(0, min(3, total_samples - train_samples - test_samples), test_samples // 4):
            train_start = offset
            train_end = train_start + train_samples
            test_start = train_end
            test_end = test_start + test_samples
            
            if test_end > total_samples:
                break
            
            # Get train/test data
            X_train = data.iloc[train_start:train_end][feature_cols]
            y_train = labels.iloc[train_start:train_end]
            X_test = data.iloc[test_start:test_end][feature_cols]
            y_test = labels.iloc[test_start:test_end]
            
            # Remove NaN labels
            train_mask = ~pd.isna(y_train)
            test_mask = ~pd.isna(y_test)
            
            if train_mask.sum() < 100 or test_mask.sum() < 10:
                continue
            
            X_train_clean = X_train[train_mask]
            y_train_clean = y_train[train_mask]
            X_test_clean = X_test[test_mask]
            y_test_clean = y_test[test_mask]
            
            # Handle NaN values in features
            X_train_clean = X_train_clean.fillna(X_train_clean.mean())
            X_test_clean = X_test_clean.fillna(X_train_clean.mean())  # Use train mean for test
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)
            
            # Train model
            model.fit(X_train_scaled, y_train_clean)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Generate trades
            test_data = data.iloc[test_start:test_end][test_mask]
            
            for i, (idx, row) in enumerate(test_data.iterrows()):
                if i >= len(y_pred_proba):
                    break
                
                # Get prediction probabilities
                proba = y_pred_proba[i]
                prob_bear = proba[0] if len(proba) > 0 else 0  # class -1
                prob_flat = proba[1] if len(proba) > 1 else 0  # class 0
                prob_bull = proba[2] if len(proba) > 2 else 0  # class 1
                
                # Position sizing logic
                position = 0
                if prob_bull > 0.45 and prob_bull > max(prob_bear, prob_flat):
                    position = 1  # Long
                elif prob_bear > 0.45 and prob_bear > max(prob_bull, prob_flat):
                    position = -1  # Short
                
                # Risk filters
                if row['Current_VIX'] > self.vol_filter_threshold:
                    position = 0  # Skip trade due to high volatility
                
                # Calculate trade P&L (simplified)
                if position != 0:
                    # Simulate holding for 15 minutes
                    entry_price = row['Current_Close']
                    future_idx = min(i + 3, len(test_data) - 1)
                    if future_idx < len(test_data):
                        exit_price = test_data.iloc[future_idx]['Current_Close']
                        pnl = position * (exit_price - entry_price) / entry_price
                        
                        # Apply stop loss
                        stop_loss = row['Current_ATR'] / entry_price if row['Current_ATR'] > 0 else 0.01
                        if abs(pnl) > stop_loss:
                            pnl = -stop_loss if pnl < 0 else stop_loss
                        
                        trades.append({
                            'timestamp': idx,
                            'position': position,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'pnl': pnl,
                            'prob_bull': prob_bull,
                            'prob_bear': prob_bear,
                            'prob_flat': prob_flat,
                            'actual_label': y_test_clean.iloc[i] if i < len(y_test_clean) else 0
                        })
            
            # Store period results
            if len(y_pred) > 0:
                accuracy = balanced_accuracy_score(y_test_clean, y_pred)
                results.append({
                    'period': offset,
                    'accuracy': accuracy,
                    'n_trades': len([t for t in trades if t['timestamp'] >= data.index[test_start]]),
                    'n_samples': len(y_test_clean)
                })
        
        return {
            'results': results,
            'trades': trades,
            'model': model
        }
    
    def risk_metrics(self, trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive risk and performance metrics
        """
        if not trades:
            return {'error': 'No trades to analyze'}
        
        trades_df = pd.DataFrame(trades)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        cumulative_pnl = trades_df['pnl'].cumsum()
        max_drawdown = (cumulative_pnl - cumulative_pnl.expanding().max()).min()
        
        # Sharpe ratio (assuming daily returns)
        returns = trades_df['pnl'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 and np.std(negative_returns) > 0 else 0
        
        # Position-specific metrics
        long_trades = trades_df[trades_df['position'] == 1]
        short_trades = trades_df[trades_df['position'] == -1]
        
        long_hit_rate = len(long_trades[long_trades['pnl'] > 0]) / len(long_trades) if len(long_trades) > 0 else 0
        short_hit_rate = len(short_trades[short_trades['pnl'] > 0]) / len(short_trades) if len(short_trades) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'long_hit_rate': long_hit_rate,
            'short_hit_rate': short_hit_rate,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'cumulative_pnl': cumulative_pnl.values
        }
    
    def plot_curves(self, trades: List[Dict], metrics: Dict):
        """
        Generate performance plots
        """
        if not trades:
            print("No trades to plot")
            return
        
        trades_df = pd.DataFrame(trades)
        
        # Set up the plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[0, 0].plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        axes[0, 0].set_title('Cumulative P&L')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Cumulative P&L')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # P&L distribution
        axes[1, 0].hist(trades_df['pnl'], bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('P&L Distribution')
        axes[1, 0].set_xlabel('P&L per Trade')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Position analysis
        position_pnl = trades_df.groupby('position')['pnl'].sum()
        axes[1, 1].bar(position_pnl.index, position_pnl.values, alpha=0.7)
        axes[1, 1].set_title('P&L by Position Type')
        axes[1, 1].set_xlabel('Position (-1: Short, 1: Long)')
        axes[1, 1].set_ylabel('Total P&L')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figs/performance_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confusion matrix (if we have actual labels)
        if 'actual_label' in trades_df.columns:
            y_true = trades_df['actual_label'].values
            y_pred = trades_df['position'].values
            
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Short', 'Flat', 'Long'],
                       yticklabels=['Bear', 'Flat', 'Bull'])
            plt.title('Confusion Matrix: Actual vs Predicted')
            plt.xlabel('Predicted Position')
            plt.ylabel('Actual Label')
            plt.savefig('./figs/confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def main(self) -> Dict:
        """
        Main execution function
        """
        print("=" * 50)
        print("Multi-Asset Trading Classification System")
        print("=" * 50)
        
        try:
            # Load data
            data = self.load_data()
            
            # Build features
            features = self.build_features(data)
            
            # Create labels
            labels = self.make_labels(features)
            
            # Train model
            model = self.train_model(features, labels)
            
            # Walk-forward backtest
            backtest_results = self.walk_forward_backtest(features, labels, model)
            
            # Calculate metrics
            metrics = self.risk_metrics(backtest_results['trades'])
            
            # Print performance report
            print("\n" + "=" * 50)
            print("PERFORMANCE REPORT")
            print("=" * 50)
            
            if 'error' not in metrics:
                print(f"Total Trades: {metrics['total_trades']}")
                print(f"Winning Trades: {metrics['winning_trades']}")
                print(f"Losing Trades: {metrics['losing_trades']}")
                print(f"Win Rate: {metrics['win_rate']:.2%}")
                print(f"Total P&L: {metrics['total_pnl']:.4f}")
                print(f"Average P&L per Trade: {metrics['avg_pnl']:.4f}")
                print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
                print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
                print(f"Long Hit Rate: {metrics['long_hit_rate']:.2%}")
                print(f"Short Hit Rate: {metrics['short_hit_rate']:.2%}")
                print(f"Long Trades: {metrics['long_trades']}")
                print(f"Short Trades: {metrics['short_trades']}")
                
                # Model performance
                if backtest_results['results']:
                    avg_accuracy = np.mean([r['accuracy'] for r in backtest_results['results']])
                    print(f"Average Balanced Accuracy: {avg_accuracy:.4f}")
                
                # Generate plots
                self.plot_curves(backtest_results['trades'], metrics)
                
            else:
                print(f"Error: {metrics['error']}")
                metrics = {'error': metrics['error']}
            
            return metrics
            
        except Exception as e:
            print(f"Error in main execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


if __name__ == "__main__":
    # Initialize and run the trading system
    classifier = TradingClassifier()
    results = classifier.main()
    
    print("\n" + "=" * 50)
    print("Execution completed!")
    print("Check ./figs directory for generated plots.")
    print("=" * 50)