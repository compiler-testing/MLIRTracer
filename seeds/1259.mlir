module {
  func.func @main(%arg0: tensor<51x32x8x80x47x48xi32>, %arg1: tensor<1x1x1x80x47x1xi32>) -> tensor<47x80x32x8x51x48xi32> {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<51x32x8x80x47x48xi32>, tensor<1x1x1x80x47x1xi32>) -> tensor<51x32x8x80x47x48xi32>
    %1 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %2 = tosa.transpose %0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<51x32x8x80x47x48xi32>) -> tensor<47x80x32x8x51x48xi32>
    return %2 : tensor<47x80x32x8x51x48xi32>
  }
}
