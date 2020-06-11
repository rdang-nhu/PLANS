(define-synthax Block
  ([(Block previous_actions perceptions)
     (let ([new_actions (Action_Block previous_actions 3)])
            (let ([new_new_actions (If new_actions perceptions)])
                          (Action_Block new_new_actions 3)))]))
