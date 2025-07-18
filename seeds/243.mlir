module {
  func.func @main(%arg0: tensor<63x11xi16>, %arg1: tensor<58x9x46x40x97xi32>, %arg2: tensor<58x1x46x40x1xi32>) -> (tensor<63x11xi16>, tensor<58x9x46x40x97xi32>) {
    %0 = "tosa.const"() {values = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 0, 1>} : (tensor<63x11xi16>) -> tensor<63x11xi16>
    %2 = tosa.minimum %arg1, %arg2 : (tensor<58x9x46x40x97xi32>, tensor<58x1x46x40x1xi32>) -> tensor<58x9x46x40x97xi32>
    %3 = tosa.minimum %2, %2 : (tensor<58x9x46x40x97xi32>, tensor<58x9x46x40x97xi32>) -> tensor<58x9x46x40x97xi32>
    return %1, %3 : tensor<63x11xi16>, tensor<58x9x46x40x97xi32>
  }
}
