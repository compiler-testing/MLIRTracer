module {
  func.func @main(%arg0: tensor<86x56x64xi8>, %arg1: tensor<3x2xi32>, %arg2: tensor<f32>) -> (tensor<f32>, tensor<308224xi8>, tensor<i1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<6xindex>} : () -> !tosa.shape<6>
    %pad_const_0 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<86x56x64xi8>, !tosa.shape<6>, tensor<1xi8>) -> tensor<86x56x64xi8>
    %1 = tosa.identity %0 : (tensor<86x56x64xi8>) -> tensor<86x56x64xi8>
    %2 = tosa.floor %arg2 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.equal %2, %2 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %r_4 = tosa.const_shape {values = dense<[ 308224 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %4 = tosa.reshape %1, %r_4 : (tensor<86x56x64xi8>, !tosa.shape<1>) -> tensor<308224xi8>
    %5 = tosa.add %3, %3 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %6 = tosa.clamp %4 {min_val = 55 : i8, max_val = 85 : i8} : (tensor<308224xi8>) -> tensor<308224xi8>
    %7 = tosa.exp %2 : (tensor<f32>) -> tensor<f32>
    %8 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %9 = tosa.transpose %6 {perms = array<i32: 0>} : (tensor<308224xi8>) -> tensor<308224xi8>
    %in_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_10 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %10 = tosa.negate %5, %in_zp_10, %out_zp_10 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    return %7, %9, %10 : tensor<f32>, tensor<308224xi8>, tensor<i1>
  }
}
