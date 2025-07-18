module {
  func.func @main(%arg0: tensor<41x78x51xi1>, %arg1: tensor<41x51x92xi1>, %arg2: tensor<f32>, %arg3: tensor<20x73x53x10xi8>, %arg4: tensor<1x1x53x1xi8>) -> (tensor<41x78x92xi1>, tensor<i1>, tensor<20x73x53x10xi8>, tensor<f32>, tensor<20x1x53x1xi8>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<41x78x51xi1>, tensor<41x51x92xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<41x78x92xi1>
    %1 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.bitwise_not %0 : (tensor<41x78x92xi1>) -> tensor<41x78x92xi1>
    %4 = tosa.greater_equal %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = tosa.logical_xor %4, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.bitwise_xor %5, %4 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = tosa.maximum %arg3, %arg4 : (tensor<20x73x53x10xi8>, tensor<1x1x53x1xi8>) -> tensor<20x73x53x10xi8>
    %8 = tosa.bitwise_xor %7, %7 : (tensor<20x73x53x10xi8>, tensor<20x73x53x10xi8>) -> tensor<20x73x53x10xi8>
    %9 = tosa.reduce_product %7 {axis = 3 : i32} : (tensor<20x73x53x10xi8>) -> tensor<20x73x53x1xi8>
    %10 = tosa.reciprocal %2 : (tensor<f32>) -> tensor<f32>
    %11 = tosa.reduce_product %9 {axis = 1 : i32} : (tensor<20x73x53x1xi8>) -> tensor<20x1x53x1xi8>
    return %3, %6, %8, %10, %11 : tensor<41x78x92xi1>, tensor<i1>, tensor<20x73x53x10xi8>, tensor<f32>, tensor<20x1x53x1xi8>
  }
}
