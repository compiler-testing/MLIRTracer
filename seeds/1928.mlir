module {
  func.func @main(%arg0: tensor<92x91x85x17xf32>, %arg1: tensor<41x84xi1>) -> (tensor<92x91x85x17xf32>, tensor<1x84xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<92x91x85x17xf32>) -> tensor<92x91x85x17xf32>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<41x84xi1>) -> tensor<1x84xi1>
    %2 = tosa.log %0 : (tensor<92x91x85x17xf32>) -> tensor<92x91x85x17xf32>
    %3 = tosa.reduce_max %1 {axis = 0 : i32} : (tensor<1x84xi1>) -> tensor<1x84xi1>
    return %2, %3 : tensor<92x91x85x17xf32>, tensor<1x84xi1>
  }
}
