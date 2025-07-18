module {
  func.func @main(%arg0: tensor<34x61x63x19x43xi16>, %arg1: tensor<58x6xf32>) -> (tensor<7x17x99674x9xi16>, tensor<522x2x1xf32>, tensor<6x58xf32>) {
    %r_0 = tosa.const_shape {values = dense<[ 7, 17, 99674, 9 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<34x61x63x19x43xi16>, !tosa.shape<4>) -> tensor<7x17x99674x9xi16>
    %1 = tosa.ceil %arg1 : (tensor<58x6xf32>) -> tensor<58x6xf32>
    %2 = tosa.sigmoid %1 : (tensor<58x6xf32>) -> tensor<58x6xf32>
    %3 = tosa.maximum %2, %2 : (tensor<58x6xf32>, tensor<58x6xf32>) -> tensor<58x6xf32>
    %4 = tosa.abs %1 : (tensor<58x6xf32>) -> tensor<58x6xf32>
    %5 = tosa.reverse %4 {axis = 1 : i32} : (tensor<58x6xf32>) -> tensor<58x6xf32>
    %t_6 = tosa.const_shape {values = dense<[ 3, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %6 = tosa.tile %5, %t_6 : (tensor<58x6xf32>, !tosa.shape<2>) -> tensor<174x6xf32>
    %r_7 = tosa.const_shape {values = dense<[ 522, 2, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %7 = tosa.reshape %6, %r_7 : (tensor<174x6xf32>, !tosa.shape<3>) -> tensor<522x2x1xf32>
    %8 = "tosa.const"() {values = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %9 = tosa.transpose %3 {perms = array<i32: 1, 0>} : (tensor<58x6xf32>) -> tensor<6x58xf32>
    return %0, %7, %9 : tensor<7x17x99674x9xi16>, tensor<522x2x1xf32>, tensor<6x58xf32>
  }
}
