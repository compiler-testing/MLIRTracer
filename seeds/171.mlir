module {
  func.func @main(%arg0: tensor<i1>, %arg1: tensor<i1>, %arg2: tensor<92x34x19x60xi32>, %arg3: tensor<92x34x76x60xi32>, %arg4: tensor<18x91x54x61xi1>) -> (tensor<i1>, tensor<92x34x95x60xi32>, tensor<18x91x54x1xi1>, tensor<92x34x95x60xi32>) {
    %0 = tosa.bitwise_xor %arg0, %arg1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %1 = tosa.concat %arg2, %arg3 {axis = 2 : i32} : (tensor<92x34x19x60xi32>, tensor<92x34x76x60xi32>) -> tensor<92x34x95x60xi32>
    %2 = tosa.intdiv %1, %1 : (tensor<92x34x95x60xi32>, tensor<92x34x95x60xi32>) -> tensor<92x34x95x60xi32>
    %3 = tosa.reduce_any %arg4 {axis = 3 : i32} : (tensor<18x91x54x61xi1>) -> tensor<18x91x54x1xi1>
    %in_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_zp_4 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = tosa.negate %1, %in_zp_4, %out_zp_4 : (tensor<92x34x95x60xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<92x34x95x60xi32>
    return %0, %2, %3, %4 : tensor<i1>, tensor<92x34x95x60xi32>, tensor<18x91x54x1xi1>, tensor<92x34x95x60xi32>
  }
}
