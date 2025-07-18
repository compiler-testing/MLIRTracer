module {
  func.func @main(%arg0: tensor<47x86x15x39x78xi16>, %arg1: tensor<24x4x13xf32>) -> (tensor<24x4x13xf32>, tensor<78x39x86x15x47xi16>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<47x86x15x39x78xi16>) -> tensor<78x39x86x15x47xi16>
    %2 = tosa.rsqrt %arg1 : (tensor<24x4x13xf32>) -> tensor<24x4x13xf32>
    %3 = tosa.minimum %2, %2 : (tensor<24x4x13xf32>, tensor<24x4x13xf32>) -> tensor<24x4x13xf32>
    %4 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<78x39x86x15x47xi16>, tensor<78x39x86x15x47xi16>) -> tensor<78x39x86x15x47xi16>
    return %3, %4 : tensor<24x4x13xf32>, tensor<78x39x86x15x47xi16>
  }
}
