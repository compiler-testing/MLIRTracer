module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<46x53xi32>) -> (tensor<i1>, tensor<7x6xi32>, tensor<46x1xi1>, tensor<7x6xi1>, tensor<7x6xi32>, tensor<f32>, tensor<46x1xi1>) {
    %0 = tosa.log %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.greater_equal %0, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %2 = tosa.reduce_product %arg1 {axis = 1 : i32} : (tensor<46x53xi32>) -> tensor<46x1xi32>
    %s_3_start = tosa.const_shape {values = dense<[ 39, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_3_size = tosa.const_shape {values = dense<[ 7, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<46x1xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<7x6xi32>
    %4 = tosa.bitwise_and %3, %3 : (tensor<7x6xi32>, tensor<7x6xi32>) -> tensor<7x6xi32>
    %5 = tosa.intdiv %2, %2 : (tensor<46x1xi32>, tensor<46x1xi32>) -> tensor<46x1xi32>
    %6 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %7 = tosa.transpose %4 {perms = array<i32: 0, 1>} : (tensor<7x6xi32>) -> tensor<7x6xi32>
    %8 = tosa.greater %5, %5 : (tensor<46x1xi32>, tensor<46x1xi32>) -> tensor<46x1xi1>
    %9 = tosa.reduce_all %8 {axis = 1 : i32} : (tensor<46x1xi1>) -> tensor<46x1xi1>
    %10 = tosa.logical_xor %9, %8 : (tensor<46x1xi1>, tensor<46x1xi1>) -> tensor<46x1xi1>
    %11 = tosa.reduce_product %8 {axis = 1 : i32} : (tensor<46x1xi1>) -> tensor<46x1xi1>
    %12 = tosa.ceil %0 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.clz %11 : (tensor<46x1xi1>) -> tensor<46x1xi1>
    %14 = tosa.tanh %12 : (tensor<f32>) -> tensor<f32>
    %15 = tosa.greater %3, %3 : (tensor<7x6xi32>, tensor<7x6xi32>) -> tensor<7x6xi1>
    %16 = tosa.minimum %4, %3 : (tensor<7x6xi32>, tensor<7x6xi32>) -> tensor<7x6xi32>
    %17 = tosa.floor %14 : (tensor<f32>) -> tensor<f32>
    %18 = tosa.logical_xor %13, %11 : (tensor<46x1xi1>, tensor<46x1xi1>) -> tensor<46x1xi1>
    return %1, %7, %10, %15, %16, %17, %18 : tensor<i1>, tensor<7x6xi32>, tensor<46x1xi1>, tensor<7x6xi1>, tensor<7x6xi32>, tensor<f32>, tensor<46x1xi1>
  }
}
