module {
  func.func @main(%arg0: tensor<75x90x100x13x42x39xi64>, %arg1: tensor<97x21xi64>) -> (tensor<42x13x90x100x75x39xi64>, tensor<291x42xi64>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0, 5]> : tensor<6xi32>} : () -> tensor<6xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0, 5>} : (tensor<75x90x100x13x42x39xi64>) -> tensor<42x13x90x100x75x39xi64>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.tile %arg1, %t_2 : (tensor<97x21xi64>, !tosa.shape<2>) -> tensor<291x42xi64>
    return %1, %2 : tensor<42x13x90x100x75x39xi64>, tensor<291x42xi64>
  }
}
