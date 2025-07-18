module {
  func.func @main(%arg0: tensor<48x61x16x39x58xi8>, %arg1: tensor<38xf32>) -> (tensor<58x39x61x16x48xi8>, tensor<38xf32>) {
    %0 = "tosa.const"() {values = dense<[4, 3, 1, 2, 0]> : tensor<5xi32>} : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0 {perms = array<i32: 4, 3, 1, 2, 0>} : (tensor<48x61x16x39x58xi8>) -> tensor<58x39x61x16x48xi8>
    %2 = tosa.bitwise_or %1, %1 : (tensor<58x39x61x16x48xi8>, tensor<58x39x61x16x48xi8>) -> tensor<58x39x61x16x48xi8>
    %3 = tosa.exp %arg1 : (tensor<38xf32>) -> tensor<38xf32>
    %4 = tosa.rsqrt %3 : (tensor<38xf32>) -> tensor<38xf32>
    return %2, %4 : tensor<58x39x61x16x48xi8>, tensor<38xf32>
  }
}
