module {
  func.func @main(%arg0: tensor<50x43x59x88x86xi8>, %arg1: tensor<5x2xi32>, %arg2: tensor<i1>, %arg3: tensor<89x31x40x69xf32>) -> (tensor<1x1x1x1xi1>, tensor<50x43x59x88x86xi1>, tensor<50x43x59x88x86xi1>, tensor<89x31x40x69xf32>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<10xindex>} : () -> !tosa.shape<10>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<50x43x59x88x86xi8>, !tosa.shape<10>, tensor<1xi8>) -> tensor<50x43x59x88x86xi8>
    %1 = tosa.logical_not %arg2 : (tensor<i1>) -> tensor<i1>
    %r_2 = tosa.const_shape {values = dense<[ 1, 1, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.reshape %1, %r_2 : (tensor<i1>, !tosa.shape<4>) -> tensor<1x1x1x1xi1>
    %3 = tosa.reduce_min %2 {axis = 2 : i32} : (tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %4 = tosa.reverse %3 {axis = 2 : i32} : (tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %5 = tosa.maximum %0, %0 : (tensor<50x43x59x88x86xi8>, tensor<50x43x59x88x86xi8>) -> tensor<50x43x59x88x86xi8>
    %6 = tosa.reduce_max %4 {axis = 1 : i32} : (tensor<1x1x1x1xi1>) -> tensor<1x1x1x1xi1>
    %7 = tosa.greater_equal %0, %5 : (tensor<50x43x59x88x86xi8>, tensor<50x43x59x88x86xi8>) -> tensor<50x43x59x88x86xi1>
    %8 = tosa.greater %0, %5 : (tensor<50x43x59x88x86xi8>, tensor<50x43x59x88x86xi8>) -> tensor<50x43x59x88x86xi1>
    %9 = tosa.reciprocal %arg3 : (tensor<89x31x40x69xf32>) -> tensor<89x31x40x69xf32>
    %10 = tosa.reciprocal %9 : (tensor<89x31x40x69xf32>) -> tensor<89x31x40x69xf32>
    return %6, %7, %8, %10 : tensor<1x1x1x1xi1>, tensor<50x43x59x88x86xi1>, tensor<50x43x59x88x86xi1>, tensor<89x31x40x69xf32>
  }
}
