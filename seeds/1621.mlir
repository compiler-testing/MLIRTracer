module {
  func.func @main(%arg0: tensor<54x56x12x7xi16>, %arg1: tensor<46x18x8x79x79x90xi32>, %arg2: tensor<1x1x1x1x79x1xi32>, %arg3: tensor<92x3x96x13xf32>) -> (tensor<92x3x96x13xf32>, tensor<46x18x8x79x79x90xi1>, tensor<54x7x56x12xi16>) {
    %0 = "tosa.const"() {values = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<54x56x12x7xi16>) -> tensor<54x7x56x12xi16>
    %2 = tosa.bitwise_not %1 : (tensor<54x7x56x12xi16>) -> tensor<54x7x56x12xi16>
    %in_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %out_zp_3 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %3 = tosa.negate %2, %in_zp_3, %out_zp_3 : (tensor<54x7x56x12xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<54x7x56x12xi16>
    %4 = tosa.clamp %3 {min_val = 11 : i16, max_val = 126 : i16} : (tensor<54x7x56x12xi16>) -> tensor<54x7x56x12xi16>
    %5 = tosa.equal %arg1, %arg2 : (tensor<46x18x8x79x79x90xi32>, tensor<1x1x1x1x79x1xi32>) -> tensor<46x18x8x79x79x90xi1>
    %6 = tosa.exp %arg3 : (tensor<92x3x96x13xf32>) -> tensor<92x3x96x13xf32>
    %7 = tosa.clz %5 : (tensor<46x18x8x79x79x90xi1>) -> tensor<46x18x8x79x79x90xi1>
    %8 = tosa.logical_right_shift %4, %3 : (tensor<54x7x56x12xi16>, tensor<54x7x56x12xi16>) -> tensor<54x7x56x12xi16>
    return %6, %7, %8 : tensor<92x3x96x13xf32>, tensor<46x18x8x79x79x90xi1>, tensor<54x7x56x12xi16>
  }
}
