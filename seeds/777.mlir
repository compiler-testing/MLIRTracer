module {
  func.func @main(%arg0: tensor<36x36xf32>, %arg1: tensor<1x1xf32>, %arg2: tensor<26xi8>, %arg3: tensor<1xi8>, %arg4: tensor<6x79x40x12xi64>, %arg5: tensor<1x1x40x12xi64>) -> (tensor<36x1xi1>, tensor<6x79x40x12xi64>, tensor<1xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<36x36xf32>, tensor<1x1xf32>) -> tensor<36x36xi1>
    %1 = tosa.greater %arg2, %arg3 : (tensor<26xi8>, tensor<1xi8>) -> tensor<26xi1>
    %2 = tosa.reduce_max %0 {axis = 1 : i32} : (tensor<36x36xi1>) -> tensor<36x1xi1>
    %3 = tosa.reduce_any %1 {axis = 0 : i32} : (tensor<26xi1>) -> tensor<1xi1>
    %4 = tosa.minimum %arg4, %arg5 : (tensor<6x79x40x12xi64>, tensor<1x1x40x12xi64>) -> tensor<6x79x40x12xi64>
    %5 = tosa.bitwise_xor %3, %3 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %2, %4, %5 : tensor<36x1xi1>, tensor<6x79x40x12xi64>, tensor<1xi1>
  }
}
