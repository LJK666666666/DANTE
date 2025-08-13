#!/usr/bin/env python3
"""
Quick Start Script for DANTE Alloy Design Notebook

This script helps users quickly start the Jupyter notebook with proper setup.

Author: DANTE Team
Date: 2024
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path


def check_jupyter():
    """Check if Jupyter is installed."""
    try:
        result = subprocess.run(['jupyter', '--version'], 
                              capture_output=True, text=True, check=True)
        print("✅ Jupyter is installed:")
        print(f"   {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Jupyter is not installed.")
        return False


def install_jupyter():
    """Install Jupyter if not available."""
    print("📦 Installing Jupyter...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'jupyter'], 
                      check=True)
        print("✅ Jupyter installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Jupyter.")
        return False


def check_dependencies():
    """Check if required dependencies are available."""
    required_deps = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
    missing_deps = []
    
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing")
            missing_deps.append(dep)
    
    return missing_deps


def install_dependencies(missing_deps):
    """Install missing dependencies."""
    if not missing_deps:
        return True
    
    print(f"📦 Installing missing dependencies: {', '.join(missing_deps)}")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_deps, 
                      check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies.")
        return False


def start_notebook():
    """Start the Jupyter notebook."""
    notebook_path = Path("main.ipynb")
    
    if not notebook_path.exists():
        print(f"❌ Notebook file {notebook_path} not found!")
        return False
    
    print(f"🚀 Starting Jupyter notebook: {notebook_path}")
    print("📝 The notebook will open in your default web browser.")
    print("💡 If it doesn't open automatically, copy the URL from the terminal.")
    
    try:
        # Start Jupyter notebook
        subprocess.run(['jupyter', 'notebook', str(notebook_path)], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to start Jupyter notebook.")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Notebook server stopped by user.")
        return True


def main():
    """Main function."""
    print("🎯 DANTE Alloy Design - Quick Start")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not Path("main.ipynb").exists():
        print("⚠️ main.ipynb not found in current directory.")
        print("Please run this script from the src/ directory.")
        return False
    
    # Check Jupyter installation
    if not check_jupyter():
        print("\n📦 Jupyter is required to run the notebook.")
        install_choice = input("Would you like to install Jupyter? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_jupyter():
                return False
        else:
            print("❌ Cannot proceed without Jupyter.")
            print("💡 You can install it manually with: pip install jupyter")
            return False
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    missing_deps = check_dependencies()
    
    if missing_deps:
        print(f"\n📦 Missing dependencies detected: {', '.join(missing_deps)}")
        install_choice = input("Would you like to install them? (y/n): ").lower().strip()
        
        if install_choice in ['y', 'yes']:
            if not install_dependencies(missing_deps):
                print("⚠️ Some dependencies failed to install.")
                print("💡 You can install them manually with:")
                print(f"   pip install {' '.join(missing_deps)}")
        else:
            print("⚠️ Some features may not work without these dependencies.")
    
    # Test module imports
    print("\n🧪 Testing module imports...")
    try:
        from data_loader import DataLoader
        from alloy_objective import AlloyObjectiveFunction
        from neural_models import DualNetworkSurrogateModel
        from visualization import create_visualizations
        from optimization import run_simple_optimization
        from config import validate_config
        
        print("✅ All modules imported successfully!")
        
    except ImportError as e:
        print(f"⚠️ Module import warning: {e}")
        print("💡 Some features may not work, but you can still explore the notebook.")
    
    # Start the notebook
    print("\n🚀 Ready to start the notebook!")
    start_choice = input("Start Jupyter notebook now? (y/n): ").lower().strip()
    
    if start_choice in ['y', 'yes']:
        return start_notebook()
    else:
        print("💡 You can start the notebook manually with:")
        print("   jupyter notebook main.ipynb")
        return True


if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Setup completed successfully!")
            print("\n📚 Quick Tips:")
            print("   • Run cells step-by-step using Shift+Enter")
            print("   • Modify parameters to experiment")
            print("   • Check the 'figures/' directory for generated plots")
            print("   • Use 'Kernel > Restart & Run All' to run everything")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            
    except KeyboardInterrupt:
        print("\n🛑 Setup interrupted by user.")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Please check your Python environment and try again.")
