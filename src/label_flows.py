#!/usr/bin/env python3
"""
Fully vectorized flow labeling using pandas merge operations.
Fixed column name matching.
"""

import pandas as pd
import numpy as np
import os

# ---------- Load Annotations ----------

def load_annotations():
    anom_csv = 'mawi_2021/labels/20210616_anomalous_suspicious.csv'
    notice_csv = 'mawi_2021/labels/20210616_notice.csv'
    
    names_anom = ['anomalyID','srcIP','srcPort','dstIP','dstPort','taxonomy','heuristic','distance','label']
    anom_df = pd.read_csv(anom_csv, header=0, names=names_anom, skiprows=1)
    for col in ['srcIP','dstIP']:
        anom_df[col] = anom_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort']:
        anom_df[col] = pd.to_numeric(anom_df[col], errors='coerce').astype('Int64')
    anom_df = anom_df[anom_df['label'].isin(['anomalous','suspicious'])].copy()
    anom_df['anomalyID'] = anom_df['anomalyID'].astype(str)
    
    names_notice = ['srcIP','srcPort','dstIP','dstPort','taxonomy','heuristic','distance','nbDetectors','label']
    notice_df = pd.read_csv(notice_csv, header=0, names=names_notice, skiprows=1)
    for col in ['srcIP','dstIP']:
        notice_df[col] = notice_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort','nbDetectors']:
        notice_df[col] = pd.to_numeric(notice_df[col], errors='coerce').astype('Int64')
    notice_df['label'] = 'notice'
    
    return anom_df, notice_df


# ---------- Pattern-based Merge Matching ----------

def match_pattern(flows_df, filters_df, keys, can_swap_ip=False, can_swap_port=False):
    """
    Match flows against filters using pandas merge.
    Returns boolean array of length len(flows_df).
    """
    n = len(flows_df)
    matched = np.zeros(n, dtype=bool)
    
    if len(filters_df) == 0:
        return matched
    
    # Add flow index
    flows_idx = flows_df[keys + ['_idx']].copy()
    filters_idx = filters_df[keys].copy()
    
    # Direction 1: forward (flow src->dst = filter src->dst)
    merged = pd.merge(flows_idx, filters_idx, on=keys, how='inner')
    if len(merged) > 0:
        matched = np.zeros(n, dtype=bool)
        matched[merged['_idx'].values] = True
    else:
        matched = np.zeros(n, dtype=bool)
    
    # Direction 2: reverse (flow src->dst = filter dst->src)
    if can_swap_ip or can_swap_port:
        rev_flow = flows_idx.copy()
        rev_filter = filters_idx.copy()
        
        rename_map = {}
        if can_swap_ip:
            rename_map['srcIP'] = 'dstIP'
            rename_map['dstIP'] = 'srcIP'
        if can_swap_port:
            rename_map['srcPort'] = 'dstPort'
            rename_map['dstPort'] = 'srcPort'
        
        rev_flow = rev_flow.rename(columns=rename_map)
        rev_filter = rev_filter.rename(columns=rename_map)
        
        rev_merged = pd.merge(rev_flow, rev_filter, on=keys, how='inner')
        if len(rev_merged) > 0:
            matched[rev_merged['_idx'].values] = True
    
    return matched


def label_chunk_vectorized(flows_df, anom_df, notice_df):
    """Label a single chunk using vectorized merges per pattern."""
    n = len(flows_df)
    is_attack = np.zeros(n, dtype=bool)
    is_notice = np.zeros(n, dtype=bool)
    
    # Add flow index
    flows_df = flows_df.copy()
    flows_df['_idx'] = np.arange(n)
    
    # Rename flow columns to match filter columns for merging
    flow_rename = {
        'src_ip': 'srcIP',
        'dst_ip': 'dstIP', 
        'src_port': 'srcPort',
        'dst_port': 'dstPort'
    }
    flows_renamed = flows_df.rename(columns=flow_rename)
    flows_renamed['_idx'] = np.arange(n)
    
    # ---- ATTACK MATCHING ----
    anom_df = anom_df.copy()
    for col in ['srcIP','dstIP']:
        anom_df[col] = anom_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort']:
        anom_df[col] = pd.to_numeric(anom_df[col], errors='coerce').astype('Int64')
    
    anom_df['has_srcIP'] = anom_df['srcIP'].notna().astype(np.uint8)
    anom_df['has_srcPort'] = anom_df['srcPort'].notna().astype(np.uint8)
    anom_df['has_dstIP'] = anom_df['dstIP'].notna().astype(np.uint8)
    anom_df['has_dstPort'] = anom_df['dstPort'].notna().astype(np.uint8)
    anom_df['pattern'] = list(zip(anom_df['has_srcIP'], anom_df['has_srcPort'], 
                                   anom_df['has_dstIP'], anom_df['has_dstPort']))
    
    for pattern, group in anom_df.groupby('pattern'):
        keys = []
        if pattern[0]: keys.append('srcIP')
        if pattern[1]: keys.append('srcPort')
        if pattern[2]: keys.append('dstIP')
        if pattern[3]: keys.append('dstPort')
        
        if not keys:
            continue
        
        filters = group[keys].drop_duplicates()
        
        # Determine if we can swap
        can_swap_ip = ('srcIP' in keys) and ('dstIP' in keys)
        can_swap_port = ('srcPort' in keys) and ('dstPort' in keys)
        
        matched = match_pattern(flows_renamed, filters, keys, can_swap_ip, can_swap_port)
        is_attack |= matched
    
    # ---- NOTICE MATCHING ----
    notice_df = notice_df.copy()
    for col in ['srcIP','dstIP']:
        notice_df[col] = notice_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort','nbDetectors']:
        notice_df[col] = pd.to_numeric(notice_df[col], errors='coerce').astype('Int64')
    
    notice_df['has_srcIP'] = notice_df['srcIP'].notna().astype(np.uint8)
    notice_df['has_srcPort'] = notice_df['srcPort'].notna().astype(np.uint8)
    notice_df['has_dstIP'] = notice_df['dstIP'].notna().astype(np.uint8)
    notice_df['has_dstPort'] = notice_df['dstPort'].notna().astype(np.uint8)
    notice_df['pattern'] = list(zip(notice_df['has_srcIP'], notice_df['has_srcPort'], 
                                     notice_df['has_dstIP'], notice_df['has_dstPort']))
    
    for pattern, group in notice_df.groupby('pattern'):
        keys = []
        if pattern[0]: keys.append('srcIP')
        if pattern[1]: keys.append('srcPort')
        if pattern[2]: keys.append('dstIP')
        if pattern[3]: keys.append('dstPort')
        
        if not keys:
            continue
        
        filters = group[keys].drop_duplicates()
        
        can_swap_ip = ('srcIP' in keys) and ('dstIP' in keys)
        can_swap_port = ('srcPort' in keys) and ('dstPort' in keys)
        
        matched = match_pattern(flows_renamed, filters, keys, can_swap_ip, can_swap_port)
        is_notice |= matched
    
    # Combine: attack > notice > benign
    labels = np.zeros(n, dtype=np.int8)
    labels[is_attack] = 1
    labels[~is_attack & is_notice] = -1
    labels[~is_attack & ~is_notice] = 0
    
    return labels


def match_pattern(flows_df, filters_df, keys, can_swap_ip=False, can_swap_port=False):
    """
    Match flows against filters using pandas merge.
    Returns boolean array of length len(flows_df).
    """
    n = len(flows_df)
    matched = np.zeros(n, dtype=bool)
    
    if len(filters_df) == 0:
        return matched
    
    # Add flow index
    flows_idx = flows_df[keys + ['_idx']].copy()
    filters_idx = filters_df[keys].copy()
    
    # Direction 1: forward
    merged = pd.merge(flows_idx, filters_idx, on=keys, how='inner')
    if len(merged) > 0:
        matched[merged['_idx'].values] = True
    
    # Direction 2: reverse
    if can_swap_ip or can_swap_port:
        rev_flow = flows_idx.copy()
        rev_filter = filters_idx.copy()
        
        rename_map = {}
        if can_swap_ip:
            rename_map['srcIP'] = 'dstIP'
            rename_map['dstIP'] = 'srcIP'
        if can_swap_port:
            rename_map['srcPort'] = 'dstPort'
            rename_map['dstPort'] = 'srcPort'
        
        rev_flow = rev_flow.rename(columns=rename_map)
        rev_filter = rev_filter.rename(columns=rename_map)
        
        rev_merged = pd.merge(rev_flow, rev_filter, on=keys, how='inner')
        if len(rev_merged) > 0:
            matched[rev_merged['_idx'].values] = True
    
    return matched


def label_chunk_vectorized(flows_df, anom_df, notice_df):
    """Label a single chunk using vectorized merges per pattern."""
    n = len(flows_df)
    
    # Add flow index
    flows_df = flows_df.copy()
    flows_df['_idx'] = np.arange(n)
    
    # Rename flow columns to match filter columns
    flow_rename = {
        'src_ip': 'srcIP',
        'dst_ip': 'dstIP', 
        'src_port': 'srcPort',
        'dst_port': 'dstPort'
    }
    flows_renamed = flows_df.rename(columns=flow_rename)
    flows_renamed['_idx'] = np.arange(n)
    
    is_attack = np.zeros(n, dtype=bool)
    is_notice = np.zeros(n, dtype=bool)
    
    # ---- ATTACK MATCHING ----
    anom_df = anom_df.copy()
    for col in ['srcIP','dstIP']:
        anom_df[col] = anom_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort']:
        anom_df[col] = pd.to_numeric(anom_df[col], errors='coerce').astype('Int64')
    
    anom_df['has_srcIP'] = anom_df['srcIP'].notna().astype(np.uint8)
    anom_df['has_srcPort'] = anom_df['srcPort'].notna().astype(np.uint8)
    anom_df['has_dstIP'] = anom_df['dstIP'].notna().astype(np.uint8)
    anom_df['has_dstPort'] = anom_df['dstPort'].notna().astype(np.uint8)
    anom_df['pattern'] = list(zip(anom_df['has_srcIP'], anom_df['has_srcPort'], 
                                   anom_df['has_dstIP'], anom_df['has_dstPort']))
    
    for pattern, group in anom_df.groupby('pattern'):
        keys = []
        if pattern[0]: keys.append('srcIP')
        if pattern[1]: keys.append('srcPort')
        if pattern[2]: keys.append('dstIP')
        if pattern[3]: keys.append('dstPort')
        
        if not keys:
            continue
        
        filters = group[keys].drop_duplicates()
        
        can_swap_ip = ('srcIP' in keys) and ('dstIP' in keys)
        can_swap_port = ('srcPort' in keys) and ('dstPort' in keys)
        
        matched = match_pattern(flows_renamed, filters, keys, can_swap_ip, can_swap_port)
        is_attack |= matched
    
    # ---- NOTICE MATCHING ----
    notice_df = notice_df.copy()
    for col in ['srcIP','dstIP']:
        notice_df[col] = notice_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort','nbDetectors']:
        notice_df[col] = pd.to_numeric(notice_df[col], errors='coerce').astype('Int64')
    
    notice_df['has_srcIP'] = notice_df['srcIP'].notna().astype(np.uint8)
    notice_df['has_srcPort'] = notice_df['srcPort'].notna().astype(np.uint8)
    notice_df['has_dstIP'] = notice_df['dstIP'].notna().astype(np.uint8)
    notice_df['has_dstPort'] = notice_df['dstPort'].notna().astype(np.uint8)
    notice_df['pattern'] = list(zip(notice_df['has_srcIP'], notice_df['has_srcPort'], 
                                     notice_df['has_dstIP'], notice_df['has_dstPort']))
    
    for pattern, group in notice_df.groupby('pattern'):
        keys = []
        if pattern[0]: keys.append('srcIP')
        if pattern[1]: keys.append('srcPort')
        if pattern[2]: keys.append('dstIP')
        if pattern[3]: keys.append('dstPort')
        
        if not keys:
            continue
        
        filters = group[keys].drop_duplicates()
        
        can_swap_ip = ('srcIP' in keys) and ('dstIP' in keys)
        can_swap_port = ('srcPort' in keys) and ('dstPort' in keys)
        
        matched = match_pattern(flows_renamed, filters, keys, can_swap_ip, can_swap_port)
        is_notice |= matched
    
    # Combine: attack > notice > benign
    labels = np.zeros(n, dtype=np.int8)
    labels[is_attack] = 1
    labels[~is_attack & is_notice] = -1
    labels[~is_attack & ~is_notice] = 0
    
    return labels


def process_all_chunks():
    anom_df, notice_df = load_annotations()
    
    flows_dir = 'mawi_2021/flows_20210616_full'
    output_path = 'mawi_2021/flows_20210616_labeled.parquet'
    
    flow_files = sorted([f for f in os.listdir(flows_dir) if f.endswith('.parquet')])
    print(f"Total chunks: {len(flow_files)}")
    
    all_labeled = []
    total_attack = 0
    total_benign = 0
    total_notice = 0
    
    for idx, fname in enumerate(flow_files):
        df = pd.read_parquet(os.path.join(flows_dir, fname))
        
        labels = label_chunk_vectorized(df, anom_df, notice_df)
        
        df['label'] = labels
        df_labeled = df[df['label'] != -1].copy()
        all_labeled.append(df_labeled)
        
        attack_c = (labels == 1).sum()
        benign_c = (labels == 0).sum()
        notice_c = (labels == -1).sum()
        
        total_attack += attack_c
        total_benign += benign_c
        total_notice += notice_c
        
        if idx % 20 == 0 or idx == len(flow_files) - 1:
            print(f"Chunk {idx+1}/{len(flow_files)} ({fname}): Attack={attack_c}, Benign={benign_c}, Notice={notice_c}")
    
    result = pd.concat(all_labeled, ignore_index=True)
    result.to_parquet(output_path, compression='snappy', index=False)
    print(f"\n=== SUMMARY ===")
    print(f"Total labeled: {len(result)}")
    print(f"  Attack: {total_attack}")
    print(f"  Benign: {total_benign}")
    print(f"  Dropped (notice-only): {total_notice}")


def load_annotations():
    anom_csv = 'mawi_2021/labels/20210616_anomalous_suspicious.csv'
    notice_csv = 'mawi_2021/labels/20210616_notice.csv'
    
    names_anom = ['anomalyID','srcIP','srcPort','dstIP','dstPort','taxonomy','heuristic','distance','label']
    anom_df = pd.read_csv(anom_csv, header=0, names=names_anom, skiprows=1)
    for col in ['srcIP','dstIP']:
        anom_df[col] = anom_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort']:
        anom_df[col] = pd.to_numeric(anom_df[col], errors='coerce').astype('Int64')
    anom_df = anom_df[anom_df['label'].isin(['anomalous','suspicious'])].copy()
    anom_df['anomalyID'] = anom_df['anomalyID'].astype(str)
    
    names_notice = ['srcIP','srcPort','dstIP','dstPort','taxonomy','heuristic','distance','nbDetectors','label']
    notice_df = pd.read_csv(notice_csv, header=0, names=names_notice, skiprows=1)
    for col in ['srcIP','dstIP']:
        notice_df[col] = notice_df[col].replace('', np.nan)
    for col in ['srcPort','dstPort','nbDetectors']:
        notice_df[col] = pd.to_numeric(notice_df[col], errors='coerce').astype('Int64')
    notice_df['label'] = 'notice'
    
    return anom_df, notice_df


if __name__ == '__main__':
    process_all_chunks()