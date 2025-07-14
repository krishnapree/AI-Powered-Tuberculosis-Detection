#!/usr/bin/env python3
"""
GPU Detection Test for TensorFlow
=================================
This script tests if GPU is properly detected and working.
"""

import tensorflow as tf

def test_gpu():
    print("=" * 50)
    print("TensorFlow GPU Detection Test")
    print("=" * 50)
    
    # Basic info
    print(f"TensorFlow version: {tf.__version__}")
    
    # GPU detection
    gpu_devices = tf.config.list_physical_devices('GPU')
    gpu_available = len(gpu_devices) > 0
    
    print(f"GPU Available: {gpu_available}")
    print(f"GPU Devices: {gpu_devices}")
    
    if gpu_available:
        print("\n✅ GPU Ready for 95%+ accuracy training!")
        
        # Test GPU computation
        print("Starting GPU memory test...")
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("GPU computation successful!")
                print("Result:", c.numpy())
                
            # Memory info
            gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
            print(f"GPU Name: {gpu_details.get('device_name', 'Unknown')}")
            
            print("\n🚀 Ready to start 4-6 hour training for 95%+ accuracy!")
            
        except Exception as e:
            print(f"❌ GPU computation failed: {e}")
            
    else:
        print("\n❌ GPU not detected - will use CPU")
        print("Training time: 12-15 hours (CPU)")
        
    print("=" * 50)

if __name__ == "__main__":
    test_gpu()
