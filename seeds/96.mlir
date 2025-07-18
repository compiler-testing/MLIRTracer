module {
  func.func @main(%arg0: tensor<81x67x3x74x58x88xf32>, %arg1: tensor<6x2xi32>, %arg2: tensor<68x72xi1>, %arg3: tensor<48x6x5xi32>, %arg4: tensor<48x6x5xi32>) -> (tensor<48x6x5xi32>, tensor<81x67x3x74x58x88xf32>, tensor<432xi1>, tensor<1x72xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<12xindex>} : () -> !tosa.shape<12>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<81x67x3x74x58x88xf32>, !tosa.shape<12>, tensor<1xf32>) -> tensor<81x67x3x74x58x88xf32>
    %1 = tosa.ceil %0 : (tensor<81x67x3x74x58x88xf32>) -> tensor<81x67x3x74x58x88xf32>
    %in_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_2 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.negate %1, %in_zp_2, %out_zp_2 : (tensor<81x67x3x74x58x88xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<81x67x3x74x58x88xf32>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<68x72xi1>) -> tensor<1x72xi1>
    %4 = tosa.abs %3 : (tensor<1x72xi1>) -> tensor<1x72xi1>
    %t_5 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.tile %4, %t_5 : (tensor<1x72xi1>, !tosa.shape<2>) -> tensor<3x144xi1>
    %6 = tosa.reduce_min %4 {axis = 0 : i32} : (tensor<1x72xi1>) -> tensor<1x72xi1>
    %7 = tosa.intdiv %arg3, %arg4 : (tensor<48x6x5xi32>, tensor<48x6x5xi32>) -> tensor<48x6x5xi32>
    %8 = tosa.abs %6 : (tensor<1x72xi1>) -> tensor<1x72xi1>
    %9 = tosa.bitwise_or %8, %8 : (tensor<1x72xi1>, tensor<1x72xi1>) -> tensor<1x72xi1>
    %10 = tosa.log %2 : (tensor<81x67x3x74x58x88xf32>) -> tensor<81x67x3x74x58x88xf32>
    %11 = tosa.clz %9 : (tensor<1x72xi1>) -> tensor<1x72xi1>
    %r_12 = tosa.const_shape {values = dense<[ 432 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %12 = tosa.reshape %5, %r_12 : (tensor<3x144xi1>, !tosa.shape<1>) -> tensor<432xi1>
    %13 = tosa.sub %11, %9 : (tensor<1x72xi1>, tensor<1x72xi1>) -> tensor<1x72xi1>
    return %7, %10, %12, %13 : tensor<48x6x5xi32>, tensor<81x67x3x74x58x88xf32>, tensor<432xi1>, tensor<1x72xi1>
  }
}
