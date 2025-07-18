module {
  func.func @main(%arg0: tensor<23x72x92x53x10xf32>, %arg1: tensor<22x19x15x78x31xi64>, %arg2: tensor<22x19x1x1x31xi64>) -> (tensor<10x53x72x92x23xf32>, tensor<22x19x15x78x31xi64>) {
    %0 = tosa.tanh %arg0 : (tensor<23x72x92x53x10xf32>) -> tensor<23x72x92x53x10xf32>
    %1 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<23x72x92x53x10xf32>) -> tensor<10x53x72x92x23xf32>
    %3 = tosa.logical_left_shift %arg1, %arg2 : (tensor<22x19x15x78x31xi64>, tensor<22x19x1x1x31xi64>) -> tensor<22x19x15x78x31xi64>
    return %2, %3 : tensor<10x53x72x92x23xf32>, tensor<22x19x15x78x31xi64>
  }
}
