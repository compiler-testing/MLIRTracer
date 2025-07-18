module {
  func.func @main(%arg0: tensor<50x31x23x96xi32>, %arg1: tensor<99x31x23x96xi32>) -> tensor<149x31x96xi32> {
    %0 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<50x31x23x96xi32>, tensor<99x31x23x96xi32>) -> tensor<149x31x23x96xi32>
    %1 = tosa.argmax %0 {axis = 2 : i32} : (tensor<149x31x23x96xi32>) -> tensor<149x31x96xi32>
    return %1 : tensor<149x31x96xi32>
  }
}
