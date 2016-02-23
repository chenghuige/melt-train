TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
i=$1
o=${i/.cc/.so} 
g++ -std=c++11 -shared $i -o $o -I $TF_INC -l tensorflow_framework -L $TF_LIB -fPIC -Wl,-rpath $TF_LIB
