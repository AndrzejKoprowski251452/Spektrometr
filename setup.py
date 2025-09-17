#!/usr/bin/env python3
"""
Python Spektrometr - Environment Setup Script
Automatycznie konfiguruje środowisko Python dla aplikacji spektrometru.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Sprawdza wersję Pythona"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 7):
        print("❌ ERROR: Python 3.7+ is required!")
        return False
    print("✅ Python version OK")
    return True

def install_requirements():
    """Instaluje wymagane pakiety"""
    print("\n📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_pixelink():
    """Sprawdza dostępność PixeLink SDK"""
    try:
        import pixelinkWrapper
        print("✅ PixeLink SDK found")
        return True
    except ImportError:
        print("⚠️  PixeLink SDK not found - camera features will be limited")
        print("   Install PixeLink SDK separately if needed")
        return False

def create_directories():
    """Tworzy wymagane katalogi"""
    directories = [
        "measurement_data",
        "backup",
        "logs"
    ]
    
    print("\n📁 Creating directories...")
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created: {dir_name}")

def verify_installation():
    """Weryfikuje instalację"""
    print("\n🔍 Verifying installation...")
    
    required_modules = [
        'numpy', 'matplotlib', 'cv2', 'PIL', 'serial', 'json', 'tkinter'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing modules: {missing}")
        return False
    
    print("✅ All core modules available")
    return True

def main():
    """Główna funkcja setup"""
    print("=" * 50)
    print("🔬 Python Spektrometr - Environment Setup")
    print("=" * 50)
    
    # Sprawdź wersję Pythona
    if not check_python_version():
        return 1
    
    # Sprawdź czy jesteśmy w odpowiednim katalogu
    if not os.path.exists("requirements.txt"):
        print("❌ ERROR: requirements.txt not found!")
        print("   Run this script from the Spektrometr directory")
        return 1
    
    # Instaluj pakiety
    if not install_requirements():
        return 1
    
    # Utwórz katalogi
    create_directories()
    
    # Sprawdź PixeLink
    check_pixelink()
    
    # Weryfikuj instalację
    if not verify_installation():
        return 1
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("✅ You can now run: python 'index copy.py'")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit(main())