module {
  func.func @main(%arg0: tensor<88x27x68x58x56xi16>) -> tensor<56x58x27x68x88xi16> {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<88x27x68x58x56xi16>) -> tensor<56x58x27x68x88xi16>
    return %1 : tensor<56x58x27x68x88xi16>
  }
}
