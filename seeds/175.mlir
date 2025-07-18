module {
  func.func @main(%arg0: tensor<53x86x91xi1>, %arg1: tensor<1x86x91xi1>, %arg2: tensor<11x96x51x6x34xf32>, %arg3: tensor<11x1x1x6x1xf32>) -> (tensor<53x86x91xi1>, tensor<34x6x96x51x11xf32>) {
    %0 = tosa.logical_xor %arg0, %arg1 : (tensor<53x86x91xi1>, tensor<1x86x91xi1>) -> tensor<53x86x91xi1>
    %1 = tosa.pow %arg2, %arg3 : (tensor<11x96x51x6x34xf32>, tensor<11x1x1x6x1xf32>) -> tensor<11x96x51x6x34xf32>
    %2 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %3 = tosa.transpose %1 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<11x96x51x6x34xf32>) -> tensor<34x6x96x51x11xf32>
    %4 = tosa.floor %3 : (tensor<34x6x96x51x11xf32>) -> tensor<34x6x96x51x11xf32>
    %5 = tosa.logical_and %0, %0 : (tensor<53x86x91xi1>, tensor<53x86x91xi1>) -> tensor<53x86x91xi1>
    %6 = tosa.clamp %4 {min_val = -2.200000e+01 : f32, max_val = 5.700000e+01 : f32} : (tensor<34x6x96x51x11xf32>) -> tensor<34x6x96x51x11xf32>
    return %5, %6 : tensor<53x86x91xi1>, tensor<34x6x96x51x11xf32>
  }
}
