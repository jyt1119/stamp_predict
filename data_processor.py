#da'ta_processor.py：
"""
数据处理模块 - 流式/分批加载以降低内存占用
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import warnings
import gc
import shutil
warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器，负责所有数据的加载和预处理（内存友好版）"""
    
    def __init__(self, config):
        self.config = config
        self.trade_cal = None
        self.trade_dates = None
        self.basic_info = None
        self.daily_data = {}
        self.metric_data = {}
        self.moneyflow_data = {}
        
    def load_basic_info(self):
        """加载股票基本信息并过滤"""
        print("加载股票基本信息...")
        self.basic_info = pd.read_csv(self.config.BASIC_FILE, dtype={'ts_code': str})
        print(f"  总股票数: {len(self.basic_info)}")
        
        if self.config.EXCLUDE_MARKETS:
            for market in self.config.EXCLUDE_MARKETS:
                self.basic_info = self.basic_info[self.basic_info['market'] != market]
            print(f"  过滤市场后: {len(self.basic_info)}")
        
        if self.config.EXCLUDE_ST:
            self.basic_info = self.basic_info[
                ~self.basic_info['name'].str.contains('ST', na=False)
            ]
            print(f"  过滤ST后: {len(self.basic_info)}")
        
        self.basic_info['list_date'] = pd.to_datetime(
            self.basic_info['list_date'].astype(str).str.strip(),
            format='%Y%m%d', errors='coerce'
        )
        
    def load_trade_calendar(self):
        """加载交易日历"""
        print("加载交易日历...")
        self.trade_cal = pd.read_csv(self.config.TRADE_CAL_FILE)
        
        date_str = self.trade_cal['cal_date'].astype(str).str.strip()
        self.trade_cal['cal_date'] = pd.to_datetime(date_str, format='%Y%m%d', errors='coerce')
        self.trade_cal = self.trade_cal.dropna(subset=['cal_date'])
        self.trade_cal['is_open'] = self.trade_cal['is_open'].astype(int)
        
        self.trade_dates = self.trade_cal[
            (self.trade_cal['is_open'] == 1) & 
            (self.trade_cal['exchange'].isin(['SSE', 'SZSE']))
        ]['cal_date'].sort_values().values
        
        print(f"  交易日数量: {len(self.trade_dates)}")
        if len(self.trade_dates) > 0:
            print(f"  日期范围: {pd.Timestamp(self.trade_dates[0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(self.trade_dates[-1]).strftime('%Y-%m-%d')}")
    
    def _get_date_range(self, start_date: str, end_date: str):
        """获取日期范围内的交易日列表"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        return [d for d in self.trade_dates if start <= d <= end]
    
    def load_daily_data(self, start_date: str, end_date: str):
        """
        加载日频量价数据：若配置了 MAX_STOCKS，则先统计活跃度选股再按需加载；
        否则继续使用流式临时文件策略将数据按股票拆分入内存。
        """
        print("加载日频量价数据...")
        date_range = self._get_date_range(start_date, end_date)
        directory = self.config.DAILY_DIR
        if not directory.exists():
            print(f"  错误：目录不存在 {directory}")
            return

        cols = ['ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount', 'pre_close', 'pct_chg']
        valid_codes = set(self.basic_info['ts_code'].astype(str).values)

        max_stocks = getattr(self.config, 'MAX_STOCKS', None)
        if max_stocks is not None:
            # 一次性按日读取少量列统计活跃度（使用 amount 或 vol）
            print(f"  检测到 MAX_STOCKS={max_stocks}，先统计活跃度选股...")
            from collections import defaultdict
            activity = defaultdict(float)
            use_col = 'amount'
            for date in tqdm(date_range, desc="统计活跃度"):
                file_path = directory / f"{pd.Timestamp(date).strftime('%Y%m%d')}.csv"
                if not file_path.exists():
                    continue
                try:
                    sample = pd.read_csv(file_path, nrows=1)
                    if use_col not in sample.columns:
                        # fallback to volume if amount not present
                        use_col = 'vol' if 'vol' in sample.columns else sample.columns[1]
                    # 只读取 ts_code + use_col
                    df = pd.read_csv(file_path, usecols=['ts_code', use_col], dtype={'ts_code': str}, low_memory=True)
                    # 转 float 并合并到 activity
                    df[use_col] = pd.to_numeric(df[use_col], errors='coerce').fillna(0.0).astype(np.float64)
                    grouped = df.groupby('ts_code')[use_col].sum()
                    for code, val in grouped.items():
                        activity[code] += float(val)
                    del df, grouped, sample
                    gc.collect()
                except Exception:
                    continue

            # 选 top N 并与 valid_codes 取交集
            ranked = [c for c, _ in sorted(activity.items(), key=lambda x: x[1], reverse=True)]
            top_codes = [c for c in ranked if c in valid_codes][:int(max_stocks)]
            print(f"  选中股票数: {len(top_codes)} (按活跃度)")

            # 第二遍按日读取，只保留 top_codes
            frames = []
            for date in tqdm(date_range, desc="按选中股票加载数据"):
                file_path = directory / f"{pd.Timestamp(date).strftime('%Y%m%d')}.csv"
                if not file_path.exists():
                    continue
                try:
                    sample = pd.read_csv(file_path, nrows=1)
                    available = [c for c in cols if c in sample.columns]
                    if 'ts_code' not in available:
                        continue
                    usecols = [c for c in available]  # 包含 ts_code 与价格列
                    df = pd.read_csv(file_path, usecols=usecols, dtype={'ts_code': str}, low_memory=True)
                    df = df[df['ts_code'].isin(top_codes)]
                    if df.empty:
                        del df, sample
                        continue
                    df['trade_date'] = date
                    frames.append(df)
                    del df, sample
                    gc.collect()
                except Exception:
                    continue

            if not frames:
                print("  未加载到任何量价数据（按选中股票）")
                return

            combined = pd.concat(frames, ignore_index=True)
            del frames
            gc.collect()

            # 按股票分组，逐个存入字典
            for code, group in tqdm(combined.groupby('ts_code'), desc="按股票分组（top stocks）"):
                stock_df = group.copy()
                # 日期转换与降精度
                stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
                num_cols = [c for c in stock_df.columns if c not in ('ts_code', 'trade_date')]
                for c in num_cols:
                    if stock_df[c].dtype in (np.float64, np.int64):
                        stock_df[c] = stock_df[c].astype(np.float32, copy=False)
                stock_df.sort_values('trade_date', kind='mergesort', inplace=True)
                stock_df.index = pd.RangeIndex(len(stock_df))
                if len(stock_df) >= max(60, self.config.SEQUENCE_LENGTH):
                    self.daily_data[code] = stock_df

            self.selected_stocks = set(self.daily_data.keys())
            del combined
            gc.collect()
            print(f"  成功加载 {len(self.daily_data)} 只股票的量价数据 (top stocks)")
            return

        # 否则保持原有流式按日分文件写入 tmp 再加载的逻辑（不变）
        # 原实现保留（你已有实现），此处直接调用原先的临时文件实现以避免重复粘贴
        # 为简洁起见，回退到临时文件实现：
        tmp_dir = Path(self.config.DATA_DIR) / "tmp_daily"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        processed_days = 0
        for date in tqdm(date_range, desc="按日写入临时文件"):
            date_str = pd.Timestamp(date).strftime('%Y%m%d')
            file_path = directory / f"{date_str}.csv"
            if not file_path.exists():
                continue
            try:
                sample = pd.read_csv(file_path, nrows=1)
                available = [c for c in cols if c in sample.columns]
                if not available:
                    continue
                df = pd.read_csv(file_path, usecols=available, dtype={c: np.float32 for c in available if c!='ts_code'}, low_memory=True)
                if 'ts_code' in df.columns:
                    df['ts_code'] = df['ts_code'].astype(str)
                df['trade_date'] = date
                df = df[df['ts_code'].isin(valid_codes)]
                if df.empty:
                    continue
                for code, group in df.groupby('ts_code'):
                    tmp_file = tmp_dir / f"{code}.csv"
                    header = not tmp_file.exists()
                    group.to_csv(tmp_file, index=False, mode='a', header=header)
                processed_days += 1
                del df, sample
                gc.collect()
            except Exception:
                continue

        print(f"  已处理天数: {processed_days}")
        count_loaded = 0
        for tmp_file in tqdm(list(tmp_dir.glob("*.csv")), desc="加载每只股票到内存"):
            code = tmp_file.stem
            try:
                stock_df = pd.read_csv(tmp_file, dtype={'ts_code': str})
                if 'trade_date' in stock_df.columns:
                    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
                num_cols = [c for c in stock_df.columns if c not in ('ts_code', 'trade_date')]
                for c in num_cols:
                    if stock_df[c].dtype in (np.float64, np.int64):
                        stock_df[c] = stock_df[c].astype(np.float32, copy=False)
                stock_df.sort_values('trade_date', kind='mergesort', inplace=True)
                stock_df.index = pd.RangeIndex(len(stock_df))
                if len(stock_df) >= max(60, self.config.SEQUENCE_LENGTH):
                    self.daily_data[code] = stock_df
                    count_loaded += 1
            except Exception:
                pass
            finally:
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
                gc.collect()
# 如果存在临时目录则尝试删除
        if 'tmp_dir' in locals() and isinstance(tmp_dir, Path):
            try:
                tmp_dir.rmdir()
            except Exception:
                pass

        # 打印已加载数量（优先使用 count_loaded，否则回退到 self.daily_data 长度）
        self.selected_stocks = set(self.daily_data.keys())
        if 'count_loaded' in locals():
            print(f"  成功加载 {count_loaded} 只股票的量价数据")
        else:
            print(f"  成功加载 {len(self.daily_data)} 只股票的量价数据")
    
    def load_metric_data(self, start_date: str, end_date: str):
        """
        流式加载 metric 数据（按批写入 tmp_metric/{ts_code}.csv）
        每个 batch 处理后立即将分组写入磁盘，最后逐股票读取并释放临时数据
        """
        print("加载基本面指标数据...")
        directory = self.config.METRIC_DIR
        
        if not directory.exists():
            print(f"  目录不存在: {directory}")
            return
        
        date_range = self._get_date_range(start_date, end_date)
        # 找列样例
        first_file = None
        for date in date_range:
            date_str = pd.Timestamp(date).strftime('%Y%m%d')
            file_path = directory / f"{date_str}.csv"
            if file_path.exists():
                first_file = file_path
                break
        if first_file is None:
            print("  未找到任何metric数据文件")
            return
        
        sample = pd.read_csv(first_file, nrows=1)
        all_cols = sample.columns.tolist()
        keep_cols = ['ts_code']
        useful = ['pe_ttm', 'pb', 'ps_ttm', 'total_mv', 'circ_mv', 
                  'turnover_rate', 'dv_ttm', 'volume_ratio', 'close']
        for c in useful:
            if c in all_cols:
                keep_cols.append(c)
        print(f"  保留列: {keep_cols}")
        
        tmp_dir = Path(self.config.DATA_DIR) / "tmp_metric"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = 200
        valid_codes = set(self.basic_info['ts_code'].astype(str).values)
        processed_files = 0
        selected = getattr(self, "selected_stocks", None)

        for batch_start in tqdm(range(0, len(date_range), batch_size), desc="按批写入 metric 临时文件"):
            batch_dates = date_range[batch_start:batch_start + batch_size]
            frames = []
            for date in batch_dates:
                date_str = pd.Timestamp(date).strftime('%Y%m%d')
                file_path = directory / f"{date_str}.csv"
                if not file_path.exists():
                    continue
                try:
                    sample = pd.read_csv(file_path, nrows=1)
                    available = [c for c in keep_cols if c in sample.columns]
                    if not available:
                        continue
                    df = pd.read_csv(file_path, usecols=available, dtype={c: np.float32 for c in available if c!='ts_code'}, low_memory=True)
                    if 'ts_code' in df.columns:
                        df['ts_code'] = df['ts_code'].astype(str)
                    df['trade_date'] = date
                    frames.append(df)
                except Exception:
                    continue
            if not frames:
                continue
            batch_combined = pd.concat(frames, ignore_index=True)
            # 按股票分组追加到 tmp_metric
            for code, group in batch_combined.groupby('ts_code'):
                if code not in valid_codes:
                    continue
                if selected is not None and code not in selected:
                    continue
                tmp_file = tmp_dir / f"{code}.csv"
                header = not tmp_file.exists()
                group.to_csv(tmp_file, index=False, mode='a', header=header)
            processed_files += len(frames)
            del batch_combined, frames
            gc.collect()
        
        print(f"  已处理 metric 文件数（近似）: {processed_files}")
        # 逐个读取 tmp_metric 文件构建 metric_data
        count_loaded = 0
        for tmp_file in tqdm(list(tmp_dir.glob("*.csv")), desc="合并基本面到内存"):
            code = tmp_file.stem
            if selected is not None and code not in selected:
                continue
            try:
                stock_df = pd.read_csv(tmp_file, dtype={'ts_code': str})
                if 'trade_date' in stock_df.columns:
                    stock_df['trade_date'] = pd.to_datetime(stock_df['trade_date'])
                stock_df.sort_values('trade_date', kind='mergesort', inplace=True)
                stock_df.index = pd.RangeIndex(len(stock_df))
                # 转换数值到 float32
                num_cols = [c for c in stock_df.columns if c not in ('ts_code', 'trade_date')]
                for c in num_cols:
                    if stock_df[c].dtype in (np.float64, np.int64):
                        stock_df[c] = stock_df[c].astype(np.float32, copy=False)
                self.metric_data[code] = stock_df
                count_loaded += 1
            except Exception:
                pass
            finally:
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
                gc.collect()
        
        try:
            tmp_dir.rmdir()
        except Exception:
            pass
        
        print(f"  成功加载 {count_loaded} 只股票的基本面数据")
    
    def load_moneyflow_data(self, start_date: str, end_date: str):
        """资金流向数据处理（如果内存允许，可打开；否则保持注释）"""
        print("加载资金流向数据...")
        # 建议同 metric 做流式处理；此处保留原注释/占位以便需要时实现
        return
    
    def process_all(self):
        """执行完整的数据处理流程"""
        print("=" * 60)
        print("开始数据处理流程")
        print("=" * 60)
        
        self.load_basic_info()
        self.load_trade_calendar()
        
        full_start = self.config.TRAIN_START
        full_end = self.config.VAL_END
        
        self.load_daily_data(full_start, full_end)
        self.load_metric_data(full_start, full_end)
        # self.load_moneyflow_data(full_start, full_end)
        
        print("=" * 60)
        print("数据处理完成！")
        print("=" * 60)
