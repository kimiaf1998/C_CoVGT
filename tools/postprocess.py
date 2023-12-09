class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = outputs["pred_boxes"]
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = out_bbox * scale_fct

        results = [{"bboxes": b} for b in boxes]

        return results