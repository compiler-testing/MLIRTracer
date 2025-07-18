module {
  func.func @main(%arg0: tensor<44x66x88xi8>, %arg1: tensor<58x60xf32>, %arg2: tensor<71x84x84x99xi1>) -> (tensor<44x66x88xi8>, tensor<116x120xf32>, tensor<71x84x84x99xi1>, tensor<58x60xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<44x66x88xi8>) -> tensor<44x66x88xi8>
    %1 = tosa.log %arg1 : (tensor<58x60xf32>) -> tensor<58x60xf32>
    %2 = tosa.bitwise_or %0, %0 : (tensor<44x66x88xi8>, tensor<44x66x88xi8>) -> tensor<44x66x88xi8>
    %3 = tosa.exp %1 : (tensor<58x60xf32>) -> tensor<58x60xf32>
    %4 = tosa.abs %1 : (tensor<58x60xf32>) -> tensor<58x60xf32>
    %t_5 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<58x60xf32>, !tosa.shape<2>) -> tensor<116x120xf32>
    %6 = tosa.exp %3 : (tensor<58x60xf32>) -> tensor<58x60xf32>
    %7 = tosa.logical_not %arg2 : (tensor<71x84x84x99xi1>) -> tensor<71x84x84x99xi1>
    %8 = tosa.sigmoid %6 : (tensor<58x60xf32>) -> tensor<58x60xf32>
    %9 = tosa.reverse %8 {axis = 0 : i32} : (tensor<58x60xf32>) -> tensor<58x60xf32>
    return %2, %5, %7, %9 : tensor<44x66x88xi8>, tensor<116x120xf32>, tensor<71x84x84x99xi1>, tensor<58x60xf32>
  }
}
