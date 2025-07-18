module {
  func.func @main(%arg0: tensor<49xf32>, %arg1: tensor<93x99x31x2x59xi1>, %arg2: tensor<93x91x93x53xi32>, %arg3: tensor<93x91x1x53xi32>) -> (tensor<93x99x31x2x59xi1>, tensor<147xf32>, tensor<93x91x93x53xi32>) {
    %0 = tosa.exp %arg0 : (tensor<49xf32>) -> tensor<49xf32>
    %1 = tosa.tanh %0 : (tensor<49xf32>) -> tensor<49xf32>
    %2 = tosa.sigmoid %1 : (tensor<49xf32>) -> tensor<49xf32>
    %3 = tosa.add %2, %0 : (tensor<49xf32>, tensor<49xf32>) -> tensor<49xf32>
    %4 = tosa.floor %3 : (tensor<49xf32>) -> tensor<49xf32>
    %5 = tosa.bitwise_not %arg1 : (tensor<93x99x31x2x59xi1>) -> tensor<93x99x31x2x59xi1>
    %t_6 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.tile %4, %t_6 : (tensor<49xf32>, !tosa.shape<1>) -> tensor<147xf32>
    %7 = tosa.intdiv %arg2, %arg3 : (tensor<93x91x93x53xi32>, tensor<93x91x1x53xi32>) -> tensor<93x91x93x53xi32>
    return %5, %6, %7 : tensor<93x99x31x2x59xi1>, tensor<147xf32>, tensor<93x91x93x53xi32>
  }
}
