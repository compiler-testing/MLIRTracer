module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<92x86x92x60xi8>, %arg2: tensor<92x86x1x1xi8>, %arg3: tensor<34xf32>) -> (tensor<i1>, tensor<92x1x92xi32>, tensor<92x86x92x60xi1>, tensor<10x4x2x4xi1>, tensor<92x86x92xi1>, tensor<92x86x92x60xi1>, tensor<6x10x2x8xi1>, tensor<34xi1>) {
    %0 = tosa.logical_not %arg0 : (tensor<i1>) -> tensor<i1>
    %1 = tosa.clz %0 : (tensor<i1>) -> tensor<i1>
    %2 = tosa.equal %arg1, %arg2 : (tensor<92x86x92x60xi8>, tensor<92x86x1x1xi8>) -> tensor<92x86x92x60xi1>
    %3 = tosa.logical_and %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %4 = tosa.argmax %2 {axis = 3 : i32} : (tensor<92x86x92x60xi1>) -> tensor<92x86x92xi32>
    %5 = tosa.bitwise_and %4, %4 : (tensor<92x86x92xi32>, tensor<92x86x92xi32>) -> tensor<92x86x92xi32>
    %6 = tosa.bitwise_not %2 : (tensor<92x86x92x60xi1>) -> tensor<92x86x92x60xi1>
    %7 = tosa.reduce_sum %5 {axis = 1 : i32} : (tensor<92x86x92xi32>) -> tensor<92x1x92xi32>
    %8 = tosa.floor %arg3 : (tensor<34xf32>) -> tensor<34xf32>
    %9 = tosa.bitwise_and %7, %7 : (tensor<92x1x92xi32>, tensor<92x1x92xi32>) -> tensor<92x1x92xi32>
    %10 = tosa.rsqrt %8 : (tensor<34xf32>) -> tensor<34xf32>
    %11 = tosa.sub %6, %6 : (tensor<92x86x92x60xi1>, tensor<92x86x92x60xi1>) -> tensor<92x86x92x60xi1>
    %12 = tosa.arithmetic_right_shift %11, %2 {round = true} : (tensor<92x86x92x60xi1>, tensor<92x86x92x60xi1>) -> tensor<92x86x92x60xi1>
    %s_13_start = tosa.const_shape {values = dense<[ 10, 82, 82, 56 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_13_size = tosa.const_shape {values = dense<[ 10, 4, 2, 4 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %13 = tosa.slice %6, %s_13_start, %s_13_size : (tensor<92x86x92x60xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<10x4x2x4xi1>
    %14 = tosa.equal %5, %5 : (tensor<92x86x92xi32>, tensor<92x86x92xi32>) -> tensor<92x86x92xi1>
    %15 = tosa.tanh %10 : (tensor<34xf32>) -> tensor<34xf32>
    %16 = tosa.sub %6, %2 : (tensor<92x86x92x60xi1>, tensor<92x86x92x60xi1>) -> tensor<92x86x92x60xi1>
    %17 = tosa.reduce_all %6 {axis = 3 : i32} : (tensor<92x86x92x60xi1>) -> tensor<92x86x92x1xi1>
    %s_18_start = tosa.const_shape {values = dense<[ 65, 29, 14, 0 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %s_18_size = tosa.const_shape {values = dense<[ 6, 10, 2, 8 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %18 = tosa.slice %17, %s_18_start, %s_18_size : (tensor<92x86x92x1xi1>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<6x10x2x8xi1>
    %19 = tosa.greater %15, %10 : (tensor<34xf32>, tensor<34xf32>) -> tensor<34xi1>
    return %3, %9, %12, %13, %14, %16, %18, %19 : tensor<i1>, tensor<92x1x92xi32>, tensor<92x86x92x60xi1>, tensor<10x4x2x4xi1>, tensor<92x86x92xi1>, tensor<92x86x92x60xi1>, tensor<6x10x2x8xi1>, tensor<34xi1>
  }
}
