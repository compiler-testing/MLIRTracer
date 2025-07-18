module {
  func.func @main(%arg0: tensor<87x79x80xi32>, %arg1: tensor<92x66x76x91x41xf32>) -> (tensor<237x240xi32>, tensor<92x66x76x91x41xf32>, tensor<87x79x80xi32>) {
    %0 = tosa.bitwise_not %arg0 : (tensor<87x79x80xi32>) -> tensor<87x79x80xi32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<87x79x80xi32>) -> tensor<87x79x80xi32>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<87x79x80xi32>, tensor<87x79x80xi32>) -> tensor<87x79x80xi32>
    %3 = tosa.floor %arg1 : (tensor<92x66x76x91x41xf32>) -> tensor<92x66x76x91x41xf32>
    %4 = tosa.argmax %2 {axis = 0 : i32} : (tensor<87x79x80xi32>) -> tensor<79x80xi32>
    %t_5 = tosa.const_shape {values = dense<[ 3, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<79x80xi32>, !tosa.shape<2>) -> tensor<237x240xi32>
    %6 = tosa.sigmoid %3 : (tensor<92x66x76x91x41xf32>) -> tensor<92x66x76x91x41xf32>
    %7 = tosa.arithmetic_right_shift %1, %2 {round = false} : (tensor<87x79x80xi32>, tensor<87x79x80xi32>) -> tensor<87x79x80xi32>
    %8 = tosa.add %6, %3 : (tensor<92x66x76x91x41xf32>, tensor<92x66x76x91x41xf32>) -> tensor<92x66x76x91x41xf32>
    %9 = tosa.reverse %7 {axis = 1 : i32} : (tensor<87x79x80xi32>) -> tensor<87x79x80xi32>
    return %5, %8, %9 : tensor<237x240xi32>, tensor<92x66x76x91x41xf32>, tensor<87x79x80xi32>
  }
}
