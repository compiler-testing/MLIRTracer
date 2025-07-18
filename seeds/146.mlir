module {
  func.func @main(%arg0: tensor<87x39x16xi8>, %arg1: tensor<87x16x23xi8>, %arg2: tensor<57x57x57xf32>) -> (tensor<57x57x57xf32>, tensor<8x6x6xi1>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<87x39x16xi8>, tensor<87x16x23xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<87x39x23xi8>
    %1 = tosa.add %0, %0 : (tensor<87x39x23xi8>, tensor<87x39x23xi8>) -> tensor<87x39x23xi8>
    %s_2_start = tosa.const_shape {values = dense<[ 53, 33, 4 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_2_size = tosa.const_shape {values = dense<[ 8, 6, 6 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.slice %1, %s_2_start, %s_2_size : (tensor<87x39x23xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<8x6x6xi8>
    %3 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 0, 1, 2>} : (tensor<8x6x6xi8>) -> tensor<8x6x6xi8>
    %5 = tosa.equal %4, %4 : (tensor<8x6x6xi8>, tensor<8x6x6xi8>) -> tensor<8x6x6xi1>
    %6 = tosa.log %arg2 : (tensor<57x57x57xf32>) -> tensor<57x57x57xf32>
    %7 = tosa.logical_right_shift %5, %5 : (tensor<8x6x6xi1>, tensor<8x6x6xi1>) -> tensor<8x6x6xi1>
    return %6, %7 : tensor<57x57x57xf32>, tensor<8x6x6xi1>
  }
}
