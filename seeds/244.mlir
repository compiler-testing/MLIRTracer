module {
  func.func @main(%arg0: tensor<51x38x39xi16>, %arg1: tensor<62x61x83xf32>, %arg2: tensor<45x90x88x97xi1>) -> (tensor<62x61x83xf32>, tensor<45x90x1x97xi1>, tensor<76x117xi32>) {
    %0 = tosa.argmax %arg0 {axis = 0 : i32} : (tensor<51x38x39xi16>) -> tensor<38x39xi32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<38x39xi32>) -> tensor<38x39xi32>
    %2 = tosa.tanh %arg1 : (tensor<62x61x83xf32>) -> tensor<62x61x83xf32>
    %3 = tosa.reduce_all %arg2 {axis = 2 : i32} : (tensor<45x90x88x97xi1>) -> tensor<45x90x1x97xi1>
    %t_4 = tosa.const_shape {values = dense<[ 2, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.tile %1, %t_4 : (tensor<38x39xi32>, !tosa.shape<2>) -> tensor<76x117xi32>
    return %2, %3, %4 : tensor<62x61x83xf32>, tensor<45x90x1x97xi1>, tensor<76x117xi32>
  }
}
