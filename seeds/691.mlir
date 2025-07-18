module {
  func.func @main(%arg0: tensor<65x35x33x31xi16>, %arg1: tensor<17x2x91x67xf32>) -> (tensor<195x35x99x62xi16>, tensor<17x2x91x67xf32>) {
    %t_0 = tosa.const_shape {values = dense<[ 3, 1, 3, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %0 = tosa.tile %arg0, %t_0 : (tensor<65x35x33x31xi16>, !tosa.shape<4>) -> tensor<195x35x99x62xi16>
    %1 = tosa.identity %0 : (tensor<195x35x99x62xi16>) -> tensor<195x35x99x62xi16>
    %2 = tosa.floor %arg1 : (tensor<17x2x91x67xf32>) -> tensor<17x2x91x67xf32>
    return %1, %2 : tensor<195x35x99x62xi16>, tensor<17x2x91x67xf32>
  }
}
