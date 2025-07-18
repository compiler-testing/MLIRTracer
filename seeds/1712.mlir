module {
  func.func @main(%arg0: tensor<36x49x12x22x12xi1>, %arg1: tensor<4x64x15x56x64xi64>, %arg2: tensor<1x1x1x56x64xi64>) -> (tensor<4x64x15x56x64xi64>, tensor<931392x1xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 931392, 6 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<36x49x12x22x12xi1>, !tosa.shape<2>) -> tensor<931392x6xi1>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<931392x6xi1>) -> tensor<931392x6xi1>
    %2 = tosa.reduce_max %1 {axis = 1 : i32} : (tensor<931392x6xi1>) -> tensor<931392x1xi1>
    %3 = tosa.minimum %arg1, %arg2 : (tensor<4x64x15x56x64xi64>, tensor<1x1x1x56x64xi64>) -> tensor<4x64x15x56x64xi64>
    %4 = tosa.clz %2 : (tensor<931392x1xi1>) -> tensor<931392x1xi1>
    return %3, %4 : tensor<4x64x15x56x64xi64>, tensor<931392x1xi1>
  }
}
