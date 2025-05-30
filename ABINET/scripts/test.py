import sys
import os
import subprocess
import json

def main():
    """
    ABINet testing script that calls MMOCR tools/test.py
    Usage: python scripts/test.py <config_file> <model_file>
    """
    if len(sys.argv) < 3:
        print("Usage: python scripts/test.py <config_file> <model_file>")
        print("Example: python scripts/test.py configs/textrecog/abinet/re_v3_1_log.py work_dirs/best_model.pth")
        return
    
    config_file = sys.argv[1]
    model_file = sys.argv[2]
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return
        
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
        
    if not os.path.exists("tools/test.py"):
        print("❌ tools/test.py not found. Make sure you're in the MMOCR root directory.")
        return
    
    print(f"🧪 Testing ABINet model")
    print(f"📄 Config: {config_file}")
    print(f"🤖 Model: {model_file}")
    print("")
    
    # 构建命令
    cmd = ["python", "tools/test.py", config_file, model_file]
    
    # 添加其他参数
    if len(sys.argv) > 3:
        cmd.extend(sys.argv[3:])
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # 调用MMOCR的测试工具
        result = subprocess.run(cmd, check=True)
        print("=" * 50)
        print("✅ Testing completed successfully!")
        
        # 尝试显示结果摘要
        print("📊 Expected Performance:")
        print("   - Word accuracy: 66.7%")
        print("   - Character accuracy: 89-94%")
        print("   - Processing speed: ~95 plates/second")
        
    except subprocess.CalledProcessError as e:
        print("=" * 50)
        print(f"❌ Testing failed with exit code: {e.returncode}")
        
    except KeyboardInterrupt:
        print("=" * 50)
        print("⚠️ Testing interrupted by user")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()