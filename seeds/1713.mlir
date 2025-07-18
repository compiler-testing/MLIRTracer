module {
  func.func @main(%arg0: tensor<67x79x86xf32>, %arg1: tensor<67x1x1xf32>, %arg2: tensor<4x80x23x52x84xi8>) -> (tensor<67x79x86xf32>, tensor<4x80x23x52x84xi8>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<67x79x86xf32>, tensor<67x1x1xf32>) -> tensor<67x79x86xf32>
    %1 = tosa.clz %arg2 : (tensor<4x80x23x52x84xi8>) -> tensor<4x80x23x52x84xi8>
    return %0, %1 : tensor<67x79x86xf32>, tensor<4x80x23x52x84xi8>
  }
}
