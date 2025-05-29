import sys
import os
import subprocess
import argparse

def main():
    """
    ABINet inference script that calls MMOCR tools/infer.py
    Usage: python scripts/inference.py --image path/to/plate.jpg
    """
    parser = argparse.ArgumentParser(description='ABINet License Plate Recognition')
    parser.add_argument('--image', required=True, help='Path to license plate image')
    parser.add_argument('--config', default='configs/textrecog/abinet/re_v3_1_log.py', 
                       help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/best_model.pth',
                       help='Model checkpoint')
    parser.add_argument('--out-dir', default='results/', help='Output directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
        
    if not os.path.exists(args.config):
        print(f"❌ Config not found: {args.config}")
        return
        
    if not os.path.exists(args.checkpoint):
        print(f"❌ Model checkpoint not found: {args.checkpoint}")
        return
        
    if not os.path.exists("tools/infer.py"):
        print("❌ tools/infer.py not found. Make sure you're in the MMOCR root directory.")
        return
    
    print(f"🔍 Running inference on: {args.image}")
    print(f"📄 Config: {args.config}")
    print(f"🤖 Model: {args.checkpoint}")
    print("")
    
    # 构建命令
    cmd = [
        "python", "tools/infer.py",
        args.image,
        "--rec", args.config,
        "--rec-weights", args.checkpoint,
        "--out-dir", args.out_dir
    ]
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 调用MMOCR的推理工具
        result = subprocess.run(cmd, check=True)
        print("=" * 50)
        print("✅ Inference completed successfully!")
        print(f"📁 Results saved to: {args.out_dir}")
        
    except subprocess.CalledProcessError as e:
        print("=" * 50)
        print(f"❌ Inference failed with exit code: {e.returncode}")
        
    except KeyboardInterrupt:
        print("=" * 50)
        print("⚠️ Inference interrupted by user")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()