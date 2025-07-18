module {
  func.func @main(%arg0: tensor<75x14x15xi16>, %arg1: tensor<75x14x57xi16>, %arg2: tensor<61x84x39xi64>, %arg3: tensor<61x1x1xi64>) -> (tensor<75x14x72xi16>, tensor<61x84x39xi1>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 2 : i32} : (tensor<75x14x15xi16>, tensor<75x14x57xi16>) -> tensor<75x14x72xi16>
    %1 = tosa.identity %0 : (tensor<75x14x72xi16>) -> tensor<75x14x72xi16>
    %2 = tosa.equal %arg2, %arg3 : (tensor<61x84x39xi64>, tensor<61x1x1xi64>) -> tensor<61x84x39xi1>
    return %1, %2 : tensor<75x14x72xi16>, tensor<61x84x39xi1>
  }
}
