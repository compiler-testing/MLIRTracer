module {
  func.func @main(%arg0: tensor<32x44x7x7x32xi8>, %arg1: tensor<34x92x29x81x29x21xf32>) -> (tensor<1xi1>, tensor<34x92x29x81x29x21xf32>, tensor<81x29x92x34x21x29xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 2207744 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<32x44x7x7x32xi8>, !tosa.shape<1>) -> tensor<2207744xi8>
    %1 = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<2207744xi8>) -> tensor<1xi8>
    %2 = tosa.greater %1, %1 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi1>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<1xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %4 = tosa.sigmoid %arg1 : (tensor<34x92x29x81x29x21xf32>) -> tensor<34x92x29x81x29x21xf32>
    %5 = tosa.rsqrt %4 : (tensor<34x92x29x81x29x21xf32>) -> tensor<34x92x29x81x29x21xf32>
    %6 = tosa.pow %4, %4 : (tensor<34x92x29x81x29x21xf32>, tensor<34x92x29x81x29x21xf32>) -> tensor<34x92x29x81x29x21xf32>
    %7 = tosa.minimum %5, %6 : (tensor<34x92x29x81x29x21xf32>, tensor<34x92x29x81x29x21xf32>) -> tensor<34x92x29x81x29x21xf32>
    %8 = "tosa.const"() {values = dense<[3, 4, 1, 0, 5, 2]> : tensor<6xi32>} : () -> tensor<6xi32>
    %9 = tosa.transpose %6 {perms = array<i32: 3, 2, 1, 0, 5, 4>} : (tensor<34x92x29x81x29x21xf32>) -> tensor<81x29x92x34x21x29xf32>
    %10 = tosa.greater_equal %9, %9 : (tensor<81x29x92x34x21x29xf32>, tensor<81x29x92x34x21x29xf32>) -> tensor<81x29x92x34x21x29xi1>
    return %3, %7, %10 : tensor<1xi1>, tensor<34x92x29x81x29x21xf32>, tensor<81x29x92x34x21x29xi1>
  }
}
