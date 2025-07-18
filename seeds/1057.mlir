module {
  func.func @main(%arg0: tensor<46x36x47xi64>) -> tensor<46x47x36xi64> {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<46x36x47xi64>) -> tensor<46x36x47xi64>
    %1 = "tosa.const"() {values = dense<[0, 2, 1]> : tensor<3xi32>} : () -> tensor<3xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 0, 2, 1>} : (tensor<46x36x47xi64>) -> tensor<46x47x36xi64>
    %3 = tosa.bitwise_not %2 : (tensor<46x47x36xi64>) -> tensor<46x47x36xi64>
    %4 = tosa.abs %3 : (tensor<46x47x36xi64>) -> tensor<46x47x36xi64>
    %5 = tosa.bitwise_xor %4, %4 : (tensor<46x47x36xi64>, tensor<46x47x36xi64>) -> tensor<46x47x36xi64>
    return %5 : tensor<46x47x36xi64>
  }
}
