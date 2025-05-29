import sys
import os
import subprocess

def main():
    """
    ABINet training script that calls MMOCR tools/train.py
    Usage: python scripts/train.py configs/textrecog/abinet/re_v3_1_log.py
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <config_file>")
        print("Example: python scripts/train.py configs/textrecog/abinet/re_v3_1_log.py")
        return
    
    config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return
    
    # 检查tools/train.py是否存在
    if not os.path.exists("tools/train.py"):
        print("❌ tools/train.py not found. Make sure you're in the MMOCR root directory.")
        return
    
    print(f"🚀 Starting ABINet training with config: {config_file}")
    print("📊 Training details:")
    print("   - Dataset: Malaysian License Plates")
    print("   - Epochs: 40 (best at epoch 27)")
    print("   - Batch Size: 16")
    print("   - Optimizer: AdamW (lr=1e-4, weight_decay=0.05)")
    print("   - Performance: 66.7% word accuracy, 89-94% character accuracy")
    print("")
    
    # 构建命令
    cmd = ["python", "tools/train.py", config_file]
    
    # 添加其他参数（如果需要）
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 调用MMOCR的训练工具
        result = subprocess.run(cmd, check=True)
        print("=" * 50)
        print("✅ Training completed successfully!")
        print("📁 Check work_dirs/ for trained models and logs")
        
    except subprocess.CalledProcessError as e:
        print("=" * 50)
        print(f"❌ Training failed with exit code: {e.returncode}")
        
    except KeyboardInterrupt:
        print("=" * 50)
        print("⚠️ Training interrupted by user")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()