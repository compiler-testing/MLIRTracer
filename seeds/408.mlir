module {
  func.func @main(%arg0: tensor<29x80x64xi32>, %arg1: tensor<29x44x64xi32>, %arg2: tensor<15xi1>, %arg3: tensor<f32>) -> (tensor<15xi1>, tensor<15xi1>, tensor<15xi1>, tensor<f32>, tensor<1xi1>, tensor<i32>, tensor<15xi1>, tensor<1xi1>, tensor<29x124x64xi32>, tensor<i1>, tensor<1xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 1 : i32} : (tensor<29x80x64xi32>, tensor<29x44x64xi32>) -> tensor<29x124x64xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<29x124x64xi32>, tensor<29x124x64xi32>) -> tensor<29x124x64xi32>
    %2 = tosa.logical_not %arg2 : (tensor<15xi1>) -> tensor<15xi1>
    %3 = tosa.bitwise_and %1, %0 : (tensor<29x124x64xi32>, tensor<29x124x64xi32>) -> tensor<29x124x64xi32>
    %r_4 = tosa.const_shape {values = dense<[ 15 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %2, %r_4 : (tensor<15xi1>, !tosa.shape<1>) -> tensor<15xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<1xi1>
    %6 = tosa.rsqrt %arg3 : (tensor<f32>) -> tensor<f32>
    %in_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_7 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %7 = tosa.negate %4, %in_zp_7, %out_zp_7 : (tensor<15xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<15xi1>
    %8 = tosa.bitwise_xor %4, %2 : (tensor<15xi1>, tensor<15xi1>) -> tensor<15xi1>
    %9 = tosa.reciprocal %6 : (tensor<f32>) -> tensor<f32>
    %10 = tosa.sigmoid %9 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.bitwise_or %4, %2 : (tensor<15xi1>, tensor<15xi1>) -> tensor<15xi1>
    %12 = tosa.sigmoid %10 : (tensor<f32>) -> tensor<f32>
    %13 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<1xi1>
    %14 = tosa.argmax %4 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<i32>
    %15 = tosa.clz %5 : (tensor<1xi1>) -> tensor<1xi1>
    %r_16 = tosa.const_shape {values = dense<[ 15 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %16 = tosa.reshape %4, %r_16 : (tensor<15xi1>, !tosa.shape<1>) -> tensor<15xi1>
    %17 = tosa.reciprocal %10 : (tensor<f32>) -> tensor<f32>
    %18 = tosa.logical_not %15 : (tensor<1xi1>) -> tensor<1xi1>
    %s_19_start = tosa.const_shape {values = dense<[ 8 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_19_size = tosa.const_shape {values = dense<[ 6 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %19 = tosa.slice %4, %s_19_start, %s_19_size : (tensor<15xi1>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<6xi1>
    %20 = tosa.minimum %3, %3 : (tensor<29x124x64xi32>, tensor<29x124x64xi32>) -> tensor<29x124x64xi32>
    %21 = tosa.greater %10, %17 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %22 = tosa.reduce_product %19 {axis = 0 : i32} : (tensor<6xi1>) -> tensor<1xi1>
    return %7, %8, %11, %12, %13, %14, %16, %18, %20, %21, %22 : tensor<15xi1>, tensor<15xi1>, tensor<15xi1>, tensor<f32>, tensor<1xi1>, tensor<i32>, tensor<15xi1>, tensor<1xi1>, tensor<29x124x64xi32>, tensor<i1>, tensor<1xi1>
  }
}
