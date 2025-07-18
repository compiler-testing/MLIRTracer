module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<67x29x20x16xi16>, %arg2: tensor<f32>) -> (tensor<i16>, tensor<67x87x40x2xi16>, tensor<f32>) {
    %0 = tosa.clamp %arg0 {min_val = -1 : i16, max_val = 68 : i16} : (tensor<i16>) -> tensor<i16>
    %1 = tosa.reduce_min %arg1 {axis = 3 : i32} : (tensor<67x29x20x16xi16>) -> tensor<67x29x20x1xi16>
    %t_2 = tosa.const_shape {values = dense<[ 1, 3, 2, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.tile %1, %t_2 : (tensor<67x29x20x1xi16>, !tosa.shape<4>) -> tensor<67x87x40x2xi16>
    %3 = tosa.sigmoid %arg2 : (tensor<f32>) -> tensor<f32>
    return %0, %2, %3 : tensor<i16>, tensor<67x87x40x2xi16>, tensor<f32>
  }
}
