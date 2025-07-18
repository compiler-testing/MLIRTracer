module {
  func.func @main(%arg0: tensor<100x61x88xi8>, %arg1: tensor<1x61x1xi8>) -> tensor<9x10x5xi8> {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = true} : (tensor<100x61x88xi8>, tensor<1x61x1xi8>) -> tensor<100x61x88xi8>
    %1 = tosa.arithmetic_right_shift %0, %0 {round = true} : (tensor<100x61x88xi8>, tensor<100x61x88xi8>) -> tensor<100x61x88xi8>
    %2 = tosa.identity %1 : (tensor<100x61x88xi8>) -> tensor<100x61x88xi8>
    %3 = "tosa.const"() {values = dense<[1, 2, 0]> : tensor<3xi32>} : () -> tensor<3xi32>
    %4 = tosa.transpose %2 {perms = array<i32: 1, 2, 0>} : (tensor<100x61x88xi8>) -> tensor<61x88x100xi8>
    %s_5_start = tosa.const_shape {values = dense<[ 28, 15, 29 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %s_5_size = tosa.const_shape {values = dense<[ 9, 10, 5 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %5 = tosa.slice %4, %s_5_start, %s_5_size : (tensor<61x88x100xi8>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<9x10x5xi8>
    return %5 : tensor<9x10x5xi8>
  }
}
