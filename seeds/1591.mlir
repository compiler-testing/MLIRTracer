module {
  func.func @main(%arg0: tensor<67x74x9xi32>, %arg1: tensor<1x1x1xi32>, %arg2: tensor<35x52x66xf32>) -> (tensor<67x74x9xi32>, tensor<3x1x14xf32>, tensor<6x6x14xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<67x74x9xi32>, tensor<1x1x1xi32>) -> tensor<67x74x9xi32>
    %1 = tosa.ceil %arg2 : (tensor<35x52x66xf32>) -> tensor<35x52x66xf32>
    %s_2_start = tosa.const_shape {values = dense<[ 25, 32, 33 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 3, 6, 7 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<35x52x66xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<3x6x7xf32>
    %3 = tosa.concat %2, %2 {axis = 2 : i32} : (tensor<3x6x7xf32>, tensor<3x6x7xf32>) -> tensor<3x6x14xf32>
    %4 = tosa.reverse %3 {axis = 1 : i32} : (tensor<3x6x14xf32>) -> tensor<3x6x14xf32>
    %in_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_5 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = tosa.negate %0, %in_zp_5, %out_zp_5 : (tensor<67x74x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<67x74x9xi32>
    %6 = tosa.floor %4 : (tensor<3x6x14xf32>) -> tensor<3x6x14xf32>
    %7 = tosa.reduce_product %4 {axis = 1 : i32} : (tensor<3x6x14xf32>) -> tensor<3x1x14xf32>
    %8 = tosa.concat %6, %6 {axis = 0 : i32} : (tensor<3x6x14xf32>, tensor<3x6x14xf32>) -> tensor<6x6x14xf32>
    %9 = tosa.abs %8 : (tensor<6x6x14xf32>) -> tensor<6x6x14xf32>
    %10 = tosa.add %7, %7 : (tensor<3x1x14xf32>, tensor<3x1x14xf32>) -> tensor<3x1x14xf32>
    %11 = tosa.tanh %10 : (tensor<3x1x14xf32>) -> tensor<3x1x14xf32>
    %12 = tosa.greater_equal %9, %8 : (tensor<6x6x14xf32>, tensor<6x6x14xf32>) -> tensor<6x6x14xi1>
    return %5, %11, %12 : tensor<67x74x9xi32>, tensor<3x1x14xf32>, tensor<6x6x14xi1>
  }
}
