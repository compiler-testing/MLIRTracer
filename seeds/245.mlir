module {
  func.func @main(%arg0: tensor<80x22x46xi64>, %arg1: tensor<80x46x31xi64>) -> tensor<80x22x31xi64> {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<80x22x46xi64>, tensor<80x46x31xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<80x22x31xi64>
    return %0 : tensor<80x22x31xi64>
  }
}
