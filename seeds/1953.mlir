module {
  func.func @main(%arg0: tensor<48x54x72x58x34xi1>) -> tensor<34x58x54x72x48xi1> {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<48x54x72x58x34xi1>) -> tensor<34x58x54x72x48xi1>
    %2 = tosa.clz %1 : (tensor<34x58x54x72x48xi1>) -> tensor<34x58x54x72x48xi1>
    %3 = tosa.logical_or %2, %2 : (tensor<34x58x54x72x48xi1>, tensor<34x58x54x72x48xi1>) -> tensor<34x58x54x72x48xi1>
    return %3 : tensor<34x58x54x72x48xi1>
  }
}
