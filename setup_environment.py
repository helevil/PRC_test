"""
Environment Setup Script for PRC_test Project
This script checks and installs all required packages before running the project.
プロジェクト環境準備スクリプト - 必要なすべてのパッケージをチェックしてインストール
"""

import subprocess
import sys
from importlib.util import find_spec

# Define required packages with their import names and pip names
REQUIRED_PACKAGES = {"matplotlib": "matplotlib", "pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn"}


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def print_header():
    """Print the script header"""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("🔧 PRC_test Project Environment Setup")
    print("Project: Spintronics Temporal Multiplexing (STM) Analysis")
    print(f"{'='*60}{Colors.END}\n")


def check_package_installed(package_name):
    """Check if a package is installed"""
    return find_spec(package_name) is not None


def get_package_version(package_name):
    """Get the version of an installed package"""
    try:
        module = __import__(package_name)
        if hasattr(module, "__version__"):
            return module.__version__
        return "Unknown"
    except:
        return "Unknown"


def install_package(pip_name):
    """Install a package using pip"""
    print(f"  インストール中 {pip_name}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_packages():
    """Check all required packages and install missing ones"""
    print(f"{Colors.YELLOW}必要なパッケージをチェック中...{Colors.END}\n")

    missing_packages = []
    installed_packages = []

    for import_name, pip_name in REQUIRED_PACKAGES.items():
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            status = f"{Colors.GREEN}✓ インストール済み{Colors.END}"
            installed_packages.append((import_name, version))
            print(f"  {status} {import_name:<15} (v{version})")
        else:
            status = f"{Colors.RED}✗ 見つかりません{Colors.END}"
            missing_packages.append((import_name, pip_name))
            print(f"  {status} {import_name}")

    print()

    # Install missing packages
    if missing_packages:
        print(f"{Colors.YELLOW}見つからないパッケージをインストール中...{Colors.END}\n")
        for import_name, pip_name in missing_packages:
            if install_package(pip_name):
                version = get_package_version(import_name)
                print(f"  {Colors.GREEN}✓ 正常にインストール{Colors.END} {import_name} (v{version})")
            else:
                print(f"  {Colors.RED}✗ インストール失敗{Colors.END} {import_name}")
                return False
        print()

    return True


def verify_environment():
    """Final verification of all packages"""
    print(f"{Colors.YELLOW}環境を検証中...{Colors.END}\n")

    all_ok = True
    for import_name in REQUIRED_PACKAGES.keys():
        if check_package_installed(import_name):
            version = get_package_version(import_name)
            print(f"  {Colors.GREEN}✓{Colors.END} {import_name:<15} (v{version})")
        else:
            print(f"  {Colors.RED}✗{Colors.END} {import_name:<15} (失敗)")
            all_ok = False

    print()
    return all_ok


def print_usage_info():
    """Print information about the project files"""
    print(f"{Colors.BLUE}プロジェクトファイル:{Colors.END}")
    print("  • auto_STM_batch.py    - 複数のCSVファイルのバッチ処理")
    print("  • STM.py               - 単一ファイルのSTM分析")
    print("  • STM_full.py          - 詳細な出力を伴う完全なSTM分析")
    print()


def main():
    """Main function"""
    print_header()

    # Check and install packages
    if not check_and_install_packages():
        print(f"{Colors.RED}❌ インストール失敗。インターネット接続またはpip設定を確認してください。{Colors.END}\n")
        return False

    # Verify environment
    if verify_environment():
        print(f"{Colors.GREEN}✅ 環境セットアップ完了！すべてのパッケージの準備ができました。{Colors.END}\n")
        print_usage_info()
        print(f"{Colors.BLUE}次のステップ:{Colors.END}")
        print("  1. スクリプトでINPUT_DIRとOUTPUT_DIRを構成")
        print("  2. 実行: python auto_STM_batch.py  (またはSTM.py / STM_full.py)")
        print()
        return True
    else:
        print(f"{Colors.RED}❌ 環境検証失敗。一部のパッケージが見つかりません。{Colors.END}\n")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
