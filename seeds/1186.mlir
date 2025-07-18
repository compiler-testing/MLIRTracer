module {
  func.func @main(%arg0: tensor<94x85x54x69x32x55xi64>) -> tensor<32x69x85x54x94x55xi64> {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<94x85x54x69x32x55xi64>) -> tensor<32x69x85x54x94x55xi64>
    %2 = tosa.clamp %1 {min_val = 51 : i64, max_val = 135 : i64} : (tensor<32x69x85x54x94x55xi64>) -> tensor<32x69x85x54x94x55xi64>
    return %2 : tensor<32x69x85x54x94x55xi64>
  }
}
