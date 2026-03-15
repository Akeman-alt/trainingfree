import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pad_arrays(arr_list):
    """将不同长度的轨迹数组补齐对齐，方便求全局均值"""
    max_len = max([len(a) for a in arr_list if len(a) > 0])
    padded = []
    for a in arr_list:
        if len(a) == 0:
            continue
        # 用最后一个值填充缺失的步数
        padded.append(np.pad(a, (0, max_len - len(a)), mode='edge'))
    return np.array(padded), max_len

def analyze_all_samples(subdir_name):
    # 🔍 1. 搜索所有 npz 文件
    search_pattern = os.path.join("inference_outputs", "**", subdir_name, "**", "guidance_diagnostics.npz")
    npz_files = glob.glob(search_pattern, recursive=True)
    
    if not npz_files:
        print(f"❌ 未找到任何属于 '{subdir_name}' 的 .npz 文件！请检查路径。")
        return
        
    print(f"🔍 共找到 {len(npz_files)} 个样本数据，开始全局聚合分析...")
    
    # --- 宏观数据池 ---
    all_reward_curves = []
    metrics_list = []
    
    # --- 微观动力学数据池 (提取最后一轮 Iteration 的张量) ---
    all_grad_l2 = []
    all_grad_max = []
    all_vf_norm = []
    all_guid_norm = []
    t_common = None
    
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            
            # 1. 提取宏观 Reward 曲线
            iter_reward = data['iter_final_reward']
            all_reward_curves.append(iter_reward)
            
            # 2. 提取微观步长数据 (只看最后一次迭代，因为这是最终生成的轨迹)
            iters = data['iter']
            last_iter = np.max(iters)
            mask = (iters == last_iter)
            
            t_val = data['t'][mask]
            if t_common is None and len(t_val) > 0:
                t_common = t_val # 假定时间步 t 是通用的
                
            all_grad_l2.append(data['grad_l2_norm'][mask])
            all_grad_max.append(data['grad_max_norm'][mask])
            all_vf_norm.append(data['vf_update_norm'][mask])
            all_guid_norm.append(data['guidance_update_norm'][mask])
            
            # 3. 读取 CSV 评估指标
            target_dir = os.path.dirname(npz_path)
            csv_path = os.path.join(target_dir, "top_sample.csv")
            sample_data = {'npz_path': npz_path, 'final_reward': iter_reward[-1]}
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if 'bb_rmsd' in df.columns: sample_data['sc_rmsd'] = df['bb_rmsd'].iloc[0]
                if 'bb_rmsd_to_gt' in df.columns: sample_data['gt_rmsd'] = df['bb_rmsd_to_gt'].iloc[0]
                if 'inv_fold_seq_recovery' in df.columns: sample_data['seq_recovery'] = df['inv_fold_seq_recovery'].iloc[0]
            metrics_list.append(sample_data)
        except Exception as e:
            print(f"⚠️ 读取 {npz_path} 失败: {e}")
            
    df_metrics = pd.DataFrame(metrics_list)
    
    # 💾 确定保存目录
    try:
        idx = npz_files[0].split(os.sep).index(subdir_name)
        save_dir = os.sep.join(npz_files[0].split(os.sep)[:idx+1])
    except:
        save_dir = "."

    # =========================================================================
    # 📊 图 1: 全局宏观报告 (Macro Report) - 收敛与真实 RMSD 对比
    # =========================================================================
    reward_mat, max_iter_len = pad_arrays(all_reward_curves)
    mean_reward = np.mean(reward_mat, axis=0)
    std_reward = np.std(reward_mat, axis=0)
    iters_x = np.arange(max_iter_len)
    
    fig1, axs1 = plt.subplots(1, 3, figsize=(22, 6))
    
    # 1.1 收敛曲线
    axs1[0].plot(iters_x, mean_reward, 'b-', linewidth=2.5, label='Mean Reward')
    axs1[0].fill_between(iters_x, mean_reward - std_reward, mean_reward + std_reward, color='blue', alpha=0.2)
    axs1[0].set_title(f'Macro: Global Convergence (N={len(npz_files)})', fontsize=14, fontweight='bold')
    axs1[0].set_xlabel('Iteration (k)'); axs1[0].set_ylabel('Terminal Reward')
    axs1[0].grid(True, linestyle='--')
    
    # 1.2 Reward vs RMSD 散点图
    rmsd_col = 'gt_rmsd' if 'gt_rmsd' in df_metrics.columns else ('sc_rmsd' if 'sc_rmsd' in df_metrics.columns else None)
    if rmsd_col:
        valid_df = df_metrics.dropna(subset=[rmsd_col, 'final_reward'])
        sc = axs1[1].scatter(valid_df['final_reward'], valid_df[rmsd_col], c=valid_df[rmsd_col], cmap='viridis_r', s=50, alpha=0.8, edgecolor='k')
        axs1[1].set_title(f'Macro: Final Reward vs {rmsd_col.upper()}', fontsize=14, fontweight='bold')
        axs1[1].set_xlabel('Final Reward'); axs1[1].set_ylabel(f'{rmsd_col.upper()} (Å)')
        fig1.colorbar(sc, ax=axs1[1])
        axs1[1].grid(True, linestyle='--')
    
    # 1.3 成功率分布直方图
    if rmsd_col:
        axs1[2].hist(valid_df[rmsd_col], bins=20, color='coral', edgecolor='black', alpha=0.8)
        axs1[2].axvline(x=5.0, color='red', linestyle='dashed', linewidth=2, label='5.0 Å Threshold')
        axs1[2].set_title(f'Macro: {rmsd_col.upper()} Distribution', fontsize=14, fontweight='bold')
        axs1[2].set_xlabel(f'{rmsd_col.upper()} (Å)'); axs1[2].set_ylabel('Count')
        axs1[2].legend()
        axs1[2].grid(True, linestyle='--')
        
    plt.tight_layout()
    fig1_path = os.path.join(save_dir, "GLOBAL_01_macro_report.png")
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # =========================================================================
    # 🔬 图 2: 全局微观动力学报告 (Micro Dynamics) - 探究梯度是否崩溃
    # =========================================================================
    # 对齐并求平均
    mat_l2, max_t = pad_arrays(all_grad_l2)
    mat_max, _ = pad_arrays(all_grad_max)
    mat_vf, _ = pad_arrays(all_vf_norm)
    mat_guid, _ = pad_arrays(all_guid_norm)
    
    mean_l2 = np.mean(mat_l2, axis=0)
    mean_max = np.mean(mat_max, axis=0)
    mean_vf = np.mean(mat_vf, axis=0)
    mean_guid = np.mean(mat_guid, axis=0)
    
    if t_common is None or len(t_common) != max_t:
        t_common = np.linspace(1.0, 0.0, max_t) # 兜底构造时间轴
        
    fig2, axs2 = plt.subplots(1, 3, figsize=(22, 6))
    
    # 2.1 梯度范数图 (检查是否梯度爆炸/消失)
    axs2[0].plot(t_common, mean_l2, 'g-', linewidth=2.5, label='Avg Grad L2 Norm')
    axs2[0].plot(t_common, mean_max, 'r-', linewidth=2, alpha=0.7, label='Avg Grad Max Norm')
    axs2[0].set_yscale('log')
    axs2[0].invert_xaxis() # 时间从 1.0 到 0.0
    axs2[0].set_title('Micro: Gradient Magnitude over Time (t)', fontsize=14, fontweight='bold')
    axs2[0].set_xlabel('Integration Time (t) [1.0 -> 0.0]'); axs2[0].set_ylabel('Gradient Norm (Log Scale)')
    axs2[0].grid(True, linestyle='--')
    axs2[0].legend()
    
    # 2.2 主模型更新 vs 引导更新 (检查是否喧宾夺主)
    axs2[1].plot(t_common, mean_vf, color='purple', linewidth=2.5, label='Original Vector Field (VF) Update')
    axs2[1].plot(t_common, mean_guid, color='orange', linewidth=2.5, label='Guidance Update (theta)')
    axs2[1].set_yscale('log')
    axs2[1].invert_xaxis()
    axs2[1].set_title('Micro: VF Update vs Guidance Update', fontsize=14, fontweight='bold')
    axs2[1].set_xlabel('Integration Time (t) [1.0 -> 0.0]'); axs2[1].set_ylabel('Update Step Size (Log Scale)')
    axs2[1].grid(True, linestyle='--')
    axs2[1].legend()

    # 2.3 信号比例图 (SNR = Guidance / VF)
    snr = mean_guid / (mean_vf + 1e-8)
    axs2[2].plot(t_common, snr, color='teal', linewidth=2.5, label='Guidance / VF Ratio')
    axs2[2].axhline(y=1.0, color='red', linestyle='--', label='1:1 Overpower Threshold')
    axs2[2].set_yscale('log')
    axs2[2].invert_xaxis()
    axs2[2].set_title('Micro: Guidance Strength Ratio (SNR)', fontsize=14, fontweight='bold')
    axs2[2].set_xlabel('Integration Time (t) [1.0 -> 0.0]'); axs2[2].set_ylabel('Ratio (Guidance / VF)')
    axs2[2].grid(True, linestyle='--')
    axs2[2].legend()

    plt.tight_layout()
    fig2_path = os.path.join(save_dir, "GLOBAL_02_micro_dynamics.png")
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    # 保存统计 CSV
    csv_save_path = os.path.join(save_dir, "GLOBAL_metrics_summary.csv")
    df_metrics.to_csv(csv_save_path, index=False)
    
    print(f"\n✅ 分析完成！已在 {save_dir} 生成：")
    print(f"  📸 宏观收敛报告: GLOBAL_01_macro_report.png")
    print(f"  📸 微观动力学报告: GLOBAL_02_micro_dynamics.png")
    print(f"  📄 汇总表格: GLOBAL_metrics_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Globally analyze all OC-Flow samples.")
    parser.add_argument("--subdir", type=str, default="run_ocflow_distance_test", help="要聚合分析的 inference_subdir 名称")
    args = parser.parse_args()
    analyze_all_samples(args.subdir)