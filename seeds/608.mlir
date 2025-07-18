module {
  func.func @main(%arg0: tensor<65x49x92x26xi1>, %arg1: tensor<65x49x86x26xi1>, %arg2: tensor<4x5x57x29x99xi64>, %arg3: tensor<1x5x57x29x1xi64>) -> (tensor<65x49x178x26xi1>, tensor<4x5x57x29x99xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<65x49x92x26xi1>, tensor<65x49x86x26xi1>) -> tensor<65x49x178x26xi1>
    %1 = tosa.equal %arg2, %arg3 : (tensor<4x5x57x29x99xi64>, tensor<1x5x57x29x1xi64>) -> tensor<4x5x57x29x99xi1>
    %2 = tosa.logical_not %1 : (tensor<4x5x57x29x99xi1>) -> tensor<4x5x57x29x99xi1>
    %3 = tosa.sub %2, %1 : (tensor<4x5x57x29x99xi1>, tensor<4x5x57x29x99xi1>) -> tensor<4x5x57x29x99xi1>
    return %0, %3 : tensor<65x49x178x26xi1>, tensor<4x5x57x29x99xi1>
  }
}
