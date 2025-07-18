module {
  func.func @main(%arg0: tensor<91x51x77xi8>, %arg1: tensor<1x1x77xi8>, %arg2: tensor<43x54x30x58x62xf32>) -> (tensor<43x54x30x58x62xf32>, tensor<273x102x1xi1>) {
    %0 = tosa.bitwise_or %arg0, %arg1 : (tensor<91x51x77xi8>, tensor<1x1x77xi8>) -> tensor<91x51x77xi8>
    %1 = tosa.greater %0, %0 : (tensor<91x51x77xi8>, tensor<91x51x77xi8>) -> tensor<91x51x77xi1>
    %2 = tosa.reduce_all %1 {axis = 2 : i32} : (tensor<91x51x77xi1>) -> tensor<91x51x1xi1>
    %3 = tosa.reduce_all %2 {axis = 2 : i32} : (tensor<91x51x1xi1>) -> tensor<91x51x1xi1>
    %4 = tosa.floor %arg2 : (tensor<43x54x30x58x62xf32>) -> tensor<43x54x30x58x62xf32>
    %5 = tosa.log %4 : (tensor<43x54x30x58x62xf32>) -> tensor<43x54x30x58x62xf32>
    %t_6 = tosa.const_shape {values = dense<[ 3, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %6 = tosa.tile %3, %t_6 : (tensor<91x51x1xi1>, !tosa.shape<3>) -> tensor<273x102x1xi1>
    %7 = tosa.logical_not %6 : (tensor<273x102x1xi1>) -> tensor<273x102x1xi1>
    return %5, %7 : tensor<43x54x30x58x62xf32>, tensor<273x102x1xi1>
  }
}
